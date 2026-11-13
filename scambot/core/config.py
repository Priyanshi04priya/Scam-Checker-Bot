"""Schema-validated YAML configuration: routing, verdict rules, thresholds, scam patterns and
fact-check sources.

Every file is validated when it is loaded, so an invalid configuration stops the service from
starting instead of failing on the first message.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scambot.core.contracts import (
    Band,
    CheckerName,
    CheckerStatus,
    EntitySignal,
    JobKind,
    LexiconSignal,
    ScamTemplate,
    Signal,
    VerdictLabel,
)
from scambot.core.settings import CONFIG_DIR

PatternLanguage = Literal["en", "hi", "hinglish"]
PATTERN_FLAGS = re.IGNORECASE | re.DOTALL

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveSeconds = Annotated[float, Field(gt=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _require_keys(name: str, keys: object, expected: tuple[Any, ...]) -> None:
    if not isinstance(keys, Mapping):  # pragma: no cover - pydantic validates the type first
        return
    missing = [key for key in expected if key not in keys]
    if missing:
        msg = f"{name} is missing {', '.join(map(str, missing))}"
        raise ValueError(msg)


# --- routing -------------------------------------------------------------------------------


class CheckerConfig(_Config):
    timeout_s: Annotated[float, Field(gt=0, le=60)]
    depends_on: list[CheckerName] = Field(default_factory=list)
    enabled: bool = True


class RoutingConfig(_Config):
    routes: dict[JobKind, list[CheckerName]]
    checkers: dict[CheckerName, CheckerConfig]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_keys("routes", self.routes, get_args(JobKind))
        _require_keys("checkers", self.checkers, get_args(CheckerName))
        for kind, route in self.routes.items():
            if len(set(route)) != len(route):
                msg = f"route {kind} lists a checker twice"
                raise ValueError(msg)
        for name, checker in self.checkers.items():
            if name in checker.depends_on:
                msg = f"checker {name} depends on itself"
                raise ValueError(msg)
        self._check_acyclic()
        return self

    def _check_acyclic(self) -> None:
        # Kahn's algorithm: repeatedly remove checkers whose dependencies are all removed.
        pending = {name: set(cfg.depends_on) for name, cfg in self.checkers.items()}
        while pending:
            ready = [name for name, deps in pending.items() if not deps & pending.keys()]
            if not ready:
                msg = f"checker dependencies form a cycle: {', '.join(sorted(pending))}"
                raise ValueError(msg)
            for name in ready:
                del pending[name]


# --- verdict rules -------------------------------------------------------------------------


class Clause(_Config):
    checker: CheckerName
    signal: Signal
    status: CheckerStatus = "ok"
    min_confidence: Probability


class Condition(_Config):
    any_of: list[Clause] | None = None
    all_required_clear: bool = False
    always: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        chosen = [bool(self.any_of), self.all_required_clear, self.always]
        if sum(chosen) != 1:
            msg = "a rule condition needs exactly one of any_of, all_required_clear, always"
            raise ValueError(msg)
        return self


class Rule(_Config):
    id: str = Field(min_length=1)
    label: VerdictLabel
    band: Band
    when: Condition


class BandRange(_Config):
    min: Annotated[int, Field(ge=0, le=100)]
    max: Annotated[int, Field(ge=0, le=100)]
    invert: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.min > self.max:
            msg = "band min must not exceed max"
            raise ValueError(msg)
        return self


class UnknownBand(_Config):
    fixed: Annotated[int, Field(ge=0, le=100)]


class Bands(_Config):
    high: BandRange
    medium: BandRange
    low: BandRange
    unknown: UnknownBand


class VerdictRulesConfig(_Config):
    rules: list[Rule] = Field(min_length=1)
    required: dict[JobKind, list[CheckerName]]
    bands: Bands
    no_issues_requires_no_signals: bool

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [rule.id for rule in self.rules]
        if len(set(ids)) != len(ids):
            msg = "rule ids must be unique"
            raise ValueError(msg)
        always = [i for i, rule in enumerate(self.rules) if rule.when.always]
        if always != [len(self.rules) - 1]:
            msg = "exactly one rule must be 'always' and it must be the last rule"
            raise ValueError(msg)
        _require_keys("required", self.required, get_args(JobKind))
        return self


# --- thresholds ----------------------------------------------------------------------------


class PipelineThresholds(_Config):
    deadline_s: PositiveSeconds
    analyze_timeout_s: PositiveSeconds
    explanation_timeout_s: PositiveSeconds
    mcp_connect_timeout_s: PositiveSeconds
    max_text_chars: PositiveInt


class EntityThresholds(_Config):
    max_per_kind: PositiveInt
    expand_timeout_s: PositiveSeconds
    max_redirects: PositiveInt


class FuzzyThresholds(_Config):
    max_distance_short: Annotated[int, Field(ge=0)]
    max_distance_long: Annotated[int, Field(ge=0)]
    short_brand_max_len: PositiveInt
    min_token_len: PositiveInt


class ScamThresholds(_Config):
    signal_min: Probability
    llm_cap: Probability
    llm_timeout_s: PositiveSeconds
    min_span_chars: PositiveInt
    max_llm_spans: PositiveInt
    max_spans_per_name: PositiveInt
    max_text_chars: PositiveInt
    max_reputation_urls: PositiveInt
    safe_browsing_timeout_s: PositiveSeconds
    fuzzy: FuzzyThresholds


class ReplyConfig(_Config):
    summary_max_chars: PositiveInt
    max_sources: Annotated[int, Field(ge=0)]
    banned_terms: list[str]
    genuine_terms: list[str]


class Price(_Config):
    input: Annotated[Decimal, Field(ge=0)]
    output: Annotated[Decimal, Field(ge=0)]


class FactCheckThresholds(_Config):
    max_claims: PositiveInt
    max_claim_chars: PositiveInt
    top_k: PositiveInt
    rrf_k: PositiveInt
    max_candidates: PositiveInt
    api_page_size: PositiveInt
    api_timeout_s: PositiveSeconds
    index_timeout_s: PositiveSeconds
    relevance_timeout_s: PositiveSeconds
    confidence_rated: Probability
    confidence_llm_polarity: Probability
    max_evidence: PositiveInt


class IngestThresholds(_Config):
    lookback_days: PositiveInt
    max_api_pages: PositiveInt
    api_page_size: PositiveInt
    max_feed_items: PositiveInt
    fetch_timeout_s: PositiveSeconds
    max_page_bytes: PositiveInt
    per_host_delay_s: Annotated[float, Field(ge=0)]
    max_crawl_delay_s: PositiveSeconds
    embed_batch_size: PositiveInt


class OcrThresholds(_Config):
    timeout_s: PositiveSeconds
    languages: Annotated[str, Field(pattern=r"^[a-z_]+(\+[a-z_]+)*$")]
    min_word_conf: Annotated[float, Field(ge=0, le=100)]
    min_chars: PositiveInt
    upscale_below_px: PositiveInt
    max_side_px: PositiveInt
    max_video_frames: PositiveInt


class SttThresholds(_Config):
    timeout_s: PositiveSeconds
    window_s: PositiveInt
    max_input_s: PositiveInt
    beam_size: PositiveInt
    min_silence_ms: PositiveInt
    ffmpeg_timeout_s: PositiveSeconds
    min_language_prob: Probability


class KeyframeThresholds(_Config):
    timeout_s: PositiveSeconds
    max_keyframes: PositiveInt
    fallback_frames: PositiveInt
    scan_limit_s: PositiveInt
    scene_threshold: PositiveSeconds
    max_width_px: PositiveInt
    ffmpeg_timeout_s: PositiveSeconds


class MediaThresholds(_Config):
    download_timeout_s: PositiveSeconds
    preprocess_timeout_s: PositiveSeconds
    language_timeout_s: PositiveSeconds
    pool_workers: PositiveInt
    max_image_pixels: PositiveInt
    sweep_older_than_min: PositiveInt
    ocr: OcrThresholds
    stt: SttThresholds
    keyframes: KeyframeThresholds


class ThresholdsConfig(_Config):
    pipeline: PipelineThresholds
    entities: EntityThresholds
    scam: ScamThresholds
    factcheck: FactCheckThresholds
    ingest: IngestThresholds
    media: MediaThresholds
    reply: ReplyConfig
    llm_prices_usd_per_mtok: dict[str, Price]

    @model_validator(mode="after")
    def _media_fits_deadline(self) -> Self:
        media = self.media
        total = media.download_timeout_s + media.preprocess_timeout_s + media.language_timeout_s
        if total >= self.pipeline.deadline_s:
            msg = (
                "media download, preprocess and language timeouts must be below pipeline deadline_s"
            )
            raise ValueError(msg)
        return self


# --- scam patterns -------------------------------------------------------------------------


class PatternGroup(_Config):
    weight: Annotated[float, Field(gt=0, lt=1)]
    patterns: dict[PatternLanguage, list[str]]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_keys("patterns", self.patterns, get_args(PatternLanguage))
        for language, patterns in self.patterns.items():
            if not patterns:
                msg = f"no {language} patterns"
                raise ValueError(msg)
            for pattern in patterns:
                try:
                    re.compile(pattern, PATTERN_FLAGS)
                except re.error as exc:
                    msg = f"invalid {language} pattern {pattern!r}: {exc}"
                    raise ValueError(msg) from exc
        return self


class ScamPatternsConfig(_Config):
    templates: dict[ScamTemplate, PatternGroup]
    signals: dict[LexiconSignal, PatternGroup]
    entity_signals: dict[EntitySignal, Annotated[float, Field(gt=0, lt=1)]]
    shorteners: list[str] = Field(min_length=1)
    brands: dict[str, list[str]]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_keys("templates", self.templates, get_args(ScamTemplate))
        _require_keys("signals", self.signals, get_args(LexiconSignal))
        _require_keys("entity_signals", self.entity_signals, get_args(EntitySignal))
        return self


# --- fact-check sources --------------------------------------------------------------------

RatedPolarity = Literal["false", "misleading", "mixed", "true"]


def host_on_site(host: str, site: str) -> bool:
    """True when ``host`` is ``site`` itself or one of its subdomains."""
    host = host.lower().rstrip(".")
    return host == site or host.endswith("." + site)


class RatingRule(_Config):
    polarity: RatedPolarity
    terms: list[str] = Field(min_length=1)


class FactcheckSource(_Config):
    id: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    name: str = Field(min_length=1)
    site: str = Field(pattern=r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")
    language: str = Field(pattern=r"^[a-z]{2,3}$")
    api: bool = True
    feeds: list[str] = Field(default_factory=list)


class SourcesConfig(_Config):
    sources: list[FactcheckSource] = Field(min_length=1)
    ratings: list[RatingRule] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [source.id for source in self.sources]
        if len(set(ids)) != len(ids):
            msg = "source ids must be unique"
            raise ValueError(msg)
        for source in self.sources:
            for feed in source.feeds:
                parts = urlsplit(feed)
                if parts.scheme != "https" or not host_on_site(parts.hostname or "", source.site):
                    msg = f"feed of {source.id} must be an https:// URL on {source.site}"
                    raise ValueError(msg)
        polarities = [rule.polarity for rule in self.ratings]
        if sorted(polarities) != sorted(get_args(RatedPolarity)):
            msg = "ratings must list each of false, misleading, mixed and true exactly once"
            raise ValueError(msg)
        if any(not term.strip() for rule in self.ratings for term in rule.terms):
            msg = "rating terms must not be blank"
            raise ValueError(msg)
        return self

    def by_id(self, source_id: str) -> FactcheckSource | None:
        return next((source for source in self.sources if source.id == source_id), None)

    def by_host(self, host: str) -> FactcheckSource | None:
        """The source whose site serves ``host``; an ``api`` source wins over a feed-only one."""
        matches = [source for source in self.sources if host_on_site(host, source.site)]
        return next((source for source in matches if source.api), matches[0] if matches else None)


# --- loading -------------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    routing: RoutingConfig
    verdict_rules: VerdictRulesConfig
    thresholds: ThresholdsConfig
    scam_patterns: ScamPatternsConfig
    sources: SourcesConfig


def _load[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return model.model_validate(data)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        msg = f"invalid config file {path.name}: {exc}"
        raise ValueError(msg) from exc


def load_config(directory: Path = CONFIG_DIR) -> AppConfig:
    """Load and validate every behaviour configuration file in ``directory``."""
    return AppConfig(
        routing=_load(directory / "routing.yaml", RoutingConfig),
        verdict_rules=_load(directory / "verdict_rules.yaml", VerdictRulesConfig),
        thresholds=_load(directory / "thresholds.yaml", ThresholdsConfig),
        scam_patterns=_load(directory / "scam_patterns.yaml", ScamPatternsConfig),
        sources=_load(directory / "sources.yaml", SourcesConfig),
    )
