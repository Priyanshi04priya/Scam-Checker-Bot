"""Scam pattern checker: rules first, then links, then an LLM classification that can only
add to the rule score, never lower it."""

import asyncio
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

from pydantic import BaseModel

from scambot.core.config import PATTERN_FLAGS, PatternGroup, ScamPatternsConfig, ScamThresholds
from scambot.core.contracts import (
    CheckerResult,
    Entities,
    EntitySignal,
    Evidence,
    LexiconSignal,
    ScamSignal,
    ScamTemplate,
)
from scambot.core.domains import absolute_url
from scambot.core.llm import LLMClient, LLMError, LLMUnavailableError, usage_scope
from scambot.mcp_server.checkers.domains import UrlFinding, inspect_url
from scambot.mcp_server.clients.safebrowsing import SafeBrowsingError, UrlReputation

CHECKER: Final = "analyze_scam_patterns"
MAX_SPAN_CHARS: Final = 120

_TEMPLATES: Final[tuple[ScamTemplate, ...]] = get_args(ScamTemplate)
_LEXICON: Final[tuple[LexiconSignal, ...]] = get_args(LexiconSignal)
_ENTITY_SIGNALS: Final[tuple[EntitySignal, ...]] = get_args(EntitySignal)

StepStatus = Literal["ok", "skipped", "error"]


class ScamClassification(BaseModel):
    """LLM output schema; no numeric constraints (the confidence is clamped in code)."""

    template: ScamTemplate | Literal["none"]
    spans: list[str]
    confidence: float


@dataclass(frozen=True)
class _Match:
    kind: Literal["template", "signal"]
    name: str
    start: int
    end: int
    span: str


@dataclass
class _LlmOutcome:
    status: StepStatus
    template: ScamTemplate | None = None
    confidence: float = 0.0
    spans: list[str] = field(default_factory=list)
    discarded: int = 0
    score: float = 0.0
    usage: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _EntityOutcome:
    signals: dict[EntitySignal, str] = field(default_factory=dict)  # signal -> evidence span
    sources: dict[EntitySignal, str] = field(default_factory=dict)
    findings: list[UrlFinding] = field(default_factory=list)
    safe_browsing: StepStatus = "skipped"
    lookups: int = 0


def _noisy_or(weights: Sequence[float]) -> float:
    return 1.0 - math.prod(1.0 - w for w in weights)


def _clamp(value: float) -> float:
    return 0.0 if math.isnan(value) else min(max(value, 0.0), 1.0)


class ScamChecker:
    def __init__(
        self,
        patterns: ScamPatternsConfig,
        thresholds: ScamThresholds,
        llm: LLMClient,
        reputation: UrlReputation | None,
    ) -> None:
        self._config = patterns
        self._thresholds = thresholds
        self._llm = llm
        self._reputation = reputation
        self._shorteners = frozenset(patterns.shorteners)
        self._compiled: dict[str, list[re.Pattern[str]]] = {
            name: self._compile(group)
            for name, group in [*patterns.templates.items(), *patterns.signals.items()]
        }

    @staticmethod
    def _compile(group: PatternGroup) -> list[re.Pattern[str]]:
        return [re.compile(p, PATTERN_FLAGS) for ps in group.patterns.values() for p in ps]

    # --- rule pass -------------------------------------------------------------------------

    def _find(self, text: str, kind: Literal["template", "signal"], name: str) -> list[_Match]:
        found: list[_Match] = []
        limit = self._thresholds.max_spans_per_name
        for pattern in self._compiled[name]:
            for match in pattern.finditer(text):
                if match.end() > match.start():
                    found.append(_Match(kind, name, match.start(), match.end(), match.group(0)))
                if len(found) >= limit:
                    return found
        return found

    def _rule_matches(self, text: str) -> list[_Match]:
        matches = [m for name in _TEMPLATES for m in self._find(text, "template", name)]
        matches += [m for name in _LEXICON for m in self._find(text, "signal", name)]
        return matches

    # --- entity pass -----------------------------------------------------------------------

    async def _entity_signals(self, entities: Entities) -> _EntityOutcome:
        outcome = _EntityOutcome()
        fuzzy = self._thresholds.fuzzy
        checked: list[str] = []
        for entity in entities.urls:
            targets = [entity.expanded_url or entity.url]
            if entity.expanded_url and entity.expanded_url != entity.url:
                targets.append(entity.url)
            for target in targets:
                finding = inspect_url(target, self._config.brands, self._shorteners, fuzzy)
                outcome.findings.append(finding)
                if finding.is_shortener:
                    outcome.signals.setdefault("shortened_url", finding.host)
                    outcome.sources.setdefault("shortened_url", "url_check")
                if finding.is_lookalike:
                    outcome.signals.setdefault("lookalike_domain", finding.host)
                    outcome.sources.setdefault("lookalike_domain", "url_check")
            checked.append(absolute_url(targets[0]))
        if entities.phones:
            outcome.signals["phone_number"] = entities.phones[0]
            outcome.sources["phone_number"] = "rules"
        if entities.upi_ids:
            outcome.signals["upi_id"] = entities.upi_ids[0]
            outcome.sources["upi_id"] = "rules"
        await self._reputation_pass(checked, outcome)
        return outcome

    async def _reputation_pass(self, urls: list[str], outcome: _EntityOutcome) -> None:
        if self._reputation is None:
            return
        batch = list(dict.fromkeys(urls))[: self._thresholds.max_reputation_urls]
        outcome.lookups = 1 if batch else 0
        try:
            unsafe = await self._reputation.unsafe_urls(batch)
        except SafeBrowsingError:
            outcome.safe_browsing = "error"
            return
        outcome.safe_browsing = "ok"
        for url in unsafe:
            outcome.signals.setdefault("unsafe_url", url)
            outcome.sources.setdefault("unsafe_url", "safe_browsing")

    # --- LLM pass --------------------------------------------------------------------------

    async def _classify(self, text: str) -> _LlmOutcome:
        with usage_scope() as usages:
            try:
                reply = await asyncio.wait_for(
                    self._llm.structured("scam_classify", {"content": text}, ScamClassification),
                    self._thresholds.llm_timeout_s,
                )
            except LLMUnavailableError:
                outcome = _LlmOutcome(status="skipped")
            except (LLMError, TimeoutError):
                outcome = _LlmOutcome(status="error")
            else:
                outcome = self._score_llm(text, reply)
        outcome.usage = [usage.model_dump(mode="json") for usage in usages]
        return outcome

    def _score_llm(self, text: str, reply: ScamClassification) -> _LlmOutcome:
        valid: list[str] = []
        discarded = 0
        for span in reply.spans:
            if len(span.strip()) >= self._thresholds.min_span_chars and text.find(span) >= 0:
                if len(valid) < self._thresholds.max_llm_spans:
                    valid.append(span)
            else:
                discarded += 1
        template = None if reply.template == "none" else reply.template
        confidence = _clamp(reply.confidence)
        score = min(confidence, self._thresholds.llm_cap) if template and valid else 0.0
        return _LlmOutcome("ok", template, confidence, valid, discarded, score)

    # --- result ----------------------------------------------------------------------------

    async def check(
        self, text: str, entities: Entities | None, language: str | None
    ) -> CheckerResult:
        """Score ``text``; ``duration_ms`` is filled in by the tool wrapper."""
        text = text[: self._thresholds.max_text_chars]
        if not text.strip():
            return CheckerResult(
                checker=CHECKER, status="inconclusive", signal="none", confidence=0.0, duration_ms=0
            )
        matches = self._rule_matches(text)
        entity, llm = await asyncio.gather(
            self._entity_signals(entities or Entities()), self._classify(text)
        )
        return self._result(matches, entity, llm, language)

    def _result(
        self, matches: list[_Match], entity: _EntityOutcome, llm: _LlmOutcome, language: str | None
    ) -> CheckerResult:
        config = self._config
        matched_templates = [n for n in _TEMPLATES if any(m.name == n for m in matches)]
        rule_template = max(
            matched_templates, key=lambda n: config.templates[n].weight, default=None
        )
        lexicon = [n for n in _LEXICON if any(m.name == n for m in matches)]
        entity_names = [n for n in _ENTITY_SIGNALS if n in entity.signals]
        weights = [config.templates[rule_template].weight] if rule_template else []
        weights += [config.signals[n].weight for n in lexicon]
        weights += [config.entity_signals[n] for n in entity_names]
        p_rules = _noisy_or(weights)
        combined = 1.0 - (1.0 - p_rules) * (1.0 - llm.score)
        template = rule_template or (llm.template if llm.score > 0 else None)
        signals: list[ScamSignal] = [template] if template else []
        signals += [*lexicon, *entity_names]
        is_scam = combined >= self._thresholds.signal_min
        vendor_calls = {"google_safe_browsing": entity.lookups} if entity.lookups else None
        return CheckerResult(
            checker=CHECKER,
            status="ok",
            signal="scam" if is_scam else "none",
            confidence=round(combined if is_scam else 1.0 - combined, 3),
            evidence=self._evidence(matches, entity, llm, rule_template),
            duration_ms=0,
            details={
                "template": template,
                "signals": signals,
                "language": language,
                "matches": [
                    {"kind": m.kind, "name": m.name, "start": m.start, "end": m.end}
                    for m in matches
                ],
                "urls": [{"host": f.host, "flags": f.flags} for f in entity.findings],
                "safe_browsing": entity.safe_browsing,
                "llm": {
                    "status": llm.status,
                    "template": llm.template,
                    "confidence": round(llm.confidence, 3),
                    "valid_spans": len(llm.spans),
                    "discarded_spans": llm.discarded,
                },
                "rule_score": round(p_rules, 3),
                "llm_score": round(llm.score, 3),
                "combined": round(combined, 3),
                "llm_usage": llm.usage,
                **({"vendor_calls": vendor_calls} if vendor_calls else {}),
            },
        )

    @staticmethod
    def _evidence(
        matches: list[_Match],
        entity: _EntityOutcome,
        llm: _LlmOutcome,
        rule_template: ScamTemplate | None,
    ) -> list[Evidence]:
        def item(name: str, span: str, source: str) -> Evidence:
            return Evidence(summary=f'{name}: "{span[:MAX_SPAN_CHARS]}"', source=source)

        first: dict[str, _Match] = {}
        for match in matches:
            first.setdefault(match.name, match)
        evidence = [item(m.name, m.span, "rules") for m in first.values()]
        evidence += [
            item(name, entity.signals[name], entity.sources[name])
            for name in _ENTITY_SIGNALS
            if name in entity.signals
        ]
        if llm.template and llm.spans and llm.score > 0 and llm.template != rule_template:
            evidence.append(item(llm.template, llm.spans[0], "llm"))
        return evidence
