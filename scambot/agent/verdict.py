"""The verdict: priority-ordered rules over checker results. Pure and deterministic."""

import math
from collections.abc import Sequence

from scambot.core.config import BandRange, Rule, VerdictRulesConfig
from scambot.core.contracts import Band, CheckerResult, JobKind, Verdict


def band_score(band: Band, confidence: float, rules: VerdictRulesConfig) -> int:
    """Map a confidence onto the band's score range (inverted bands: confident means lower)."""
    if band == "unknown":
        return rules.bands.unknown.fixed
    span: BandRange = getattr(rules.bands, band)
    c = min(max(confidence, 0.0), 1.0)
    width = span.max - span.min
    raw = span.max - c * width if span.invert else span.min + c * width
    return min(max(math.floor(raw + 0.5), span.min), span.max)


def _match_any_of(rule: Rule, by_checker: dict[str, CheckerResult]) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for clause in rule.when.any_of or ():
        result = by_checker.get(clause.checker)
        if (
            result is not None
            and result.status == clause.status
            and result.signal == clause.signal
            and result.confidence >= clause.min_confidence
            and (best is None or result.confidence > best[1])
        ):
            best = (clause.checker, result.confidence)
    return best


def _all_required_clear(
    kind: JobKind,
    results: Sequence[CheckerResult],
    by_checker: dict[str, CheckerResult],
    rules: VerdictRulesConfig,
) -> tuple[str | None, float] | None:
    required = rules.required[kind]
    found = [by_checker.get(name) for name in required]
    if not all(r is not None and r.status == "ok" and r.signal == "none" for r in found):
        return None
    if rules.no_issues_requires_no_signals and any(
        r.status == "ok" and r.signal != "none" for r in results
    ):
        return None
    confidences = [r.confidence for r in found if r is not None]
    return (required[0] if required else None), min(confidences, default=0.0)


def _source_confirmed(by_checker: dict[str, CheckerResult]) -> bool:
    provenance = by_checker.get("check_provenance")
    caption = by_checker.get("check_caption_context")
    if provenance is None or caption is None:
        return False
    if (provenance.status, provenance.signal) != ("ok", "none"):
        return False
    origin = provenance.details.get("origin", {})
    credible = isinstance(origin, dict) and origin.get("credible") is True
    clear = (caption.status, caption.signal) == ("ok", "none")
    return credible and clear and not caption.details.get("mismatches")


def decide(results: Sequence[CheckerResult], kind: JobKind, rules: VerdictRulesConfig) -> Verdict:
    """Apply the first matching rule to ``results``."""
    by_checker = {result.checker: result for result in results}
    contributing: list[tuple[str, str, str]] = [(r.checker, r.status, r.signal) for r in results]
    unavailable = [r.checker for r in results if r.status == "error"]
    source_confirmed = _source_confirmed(by_checker)
    for rule in rules.rules:
        match: tuple[str | None, float] | None = None
        if rule.when.any_of:
            match = _match_any_of(rule, by_checker)
        elif rule.when.all_required_clear:
            match = _all_required_clear(kind, results, by_checker, rules) if results else None
        else:
            match = (None, 0.0)
        if match is None:
            continue
        deciding, confidence = match
        score = (
            rules.bands.unknown.fixed
            if rule.when.always
            else band_score(rule.band, confidence, rules)
        )
        return Verdict(
            label=rule.label,
            band=rule.band,
            score=score,
            deciding_checker=deciding,
            contributing=contributing,
            unavailable=unavailable,
            source_confirmed=source_confirmed,
        )
    msg = "verdict rules must end with an 'always' rule"  # pragma: no cover - validated config
    raise ValueError(msg)  # pragma: no cover
