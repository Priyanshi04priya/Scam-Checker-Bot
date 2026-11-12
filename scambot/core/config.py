"""Schema-validated YAML configuration: routing, verdict rules, thresholds and scam patterns.

Every file is validated when it is loaded, so an invalid configuration stops the service from
starting instead of failing on the first message.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

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


class ThresholdsConfig(_Config):
    pipeline: PipelineThresholds
    entities: EntityThresholds
    scam: ScamThresholds
    reply: ReplyConfig
    llm_prices_usd_per_mtok: dict[str, Price]


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


# --- loading -------------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    routing: RoutingConfig
    verdict_rules: VerdictRulesConfig
    thresholds: ThresholdsConfig
    scam_patterns: ScamPatternsConfig


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
    )
