"""Table-driven tests for the verdict rules."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from scambot.agent.verdict import band_score, decide
from scambot.core.config import AppConfig, VerdictRulesConfig
from scambot.core.contracts import CheckerResult, CheckerStatus, JobKind, Signal


def r(
    checker: str,
    signal: Signal = "none",
    confidence: float = 0.9,
    status: CheckerStatus = "ok",
    **details: Any,
) -> CheckerResult:
    return CheckerResult(
        checker=checker,
        status=status,
        signal=signal,
        confidence=confidence,
        duration_ms=1,
        details=details,
    )


@dataclass(frozen=True)
class Case:
    name: str
    kind: JobKind
    results: list[CheckerResult]
    label: str
    band: str
    deciding: str | None = None
    score: int | None = None
    unavailable: list[str] = field(default_factory=list)
    source_confirmed: bool = False


SCAM = "analyze_scam_patterns"
PROV = "check_provenance"
CAPTION = "check_caption_context"
FACT = "fact_check_claim"
DEEPFAKE = "detect_deepfake"
VOICE = "detect_voice_clone"

CASES = [
    # one row per rule and boundary
    Case("scam_at_threshold", "text", [r(SCAM, "scam", 0.70)], "likely_scam", "high", SCAM, 91),
    Case(
        "scam_below_threshold",
        "text",
        [r(SCAM, "scam", 0.69)],
        "could_not_verify",
        "unknown",
        score=50,
    ),
    Case(
        "old_media_at_threshold",
        "image",
        [r(PROV, "old_media", 0.60)],
        "old_or_out_of_context",
        "medium",
        PROV,
        57,
    ),
    Case("old_media_below", "image", [r(PROV, "old_media", 0.59)], "could_not_verify", "unknown"),
    Case(
        "out_of_context",
        "image",
        [r(PROV), r(CAPTION, "out_of_context", 0.6)],
        "old_or_out_of_context",
        "medium",
        CAPTION,
    ),
    Case(
        "debunked_at_threshold",
        "text",
        [r(SCAM), r(FACT, "debunked", 0.70)],
        "false_misleading",
        "high",
        FACT,
        91,
    ),
    Case(
        "debunked_below",
        "text",
        [r(SCAM, confidence=0.9), r(FACT, "debunked", 0.69)],
        "could_not_verify",
        "unknown",
    ),
    Case(
        "deepfake_at_threshold",
        "image",
        [r(DEEPFAKE, "likely_synthetic", 0.85)],
        "possibly_ai",
        "medium",
        DEEPFAKE,
    ),
    Case(
        "deepfake_below",
        "image",
        [r(DEEPFAKE, "likely_synthetic", 0.84)],
        "could_not_verify",
        "unknown",
    ),
    Case(
        "voice_at_threshold",
        "audio",
        [r(VOICE, "likely_synthetic", 0.75)],
        "possibly_ai",
        "medium",
        VOICE,
    ),
    Case(
        "voice_below", "audio", [r(VOICE, "likely_synthetic", 0.74)], "could_not_verify", "unknown"
    ),
    # scenarios
    Case(
        "old_image",
        "image",
        [r(PROV, "old_media", 0.8), r(CAPTION, "out_of_context", 0.75), r(FACT, "debunked", 0.9)],
        "old_or_out_of_context",
        "medium",
        PROV,
        63,
    ),
    Case("electricity_scam_text", "text", [r(SCAM, "scam", 0.85)], "likely_scam", "high", SCAM, 96),
    Case(
        "debunked_hindi_claim",
        "text",
        [r(SCAM, confidence=0.95), r(FACT, "debunked", 0.9)],
        "false_misleading",
        "high",
        FACT,
    ),
    Case(
        "cloned_voice_note",
        "audio",
        [r(SCAM, "scam", 0.8), r(VOICE, "likely_synthetic", 0.78)],
        "likely_scam",
        "high",
        SCAM,
    ),
    Case(
        "ai_image",
        "image",
        [r(PROV), r(DEEPFAKE, "likely_synthetic", 0.9)],
        "possibly_ai",
        "medium",
        DEEPFAKE,
    ),
    Case(
        "nothing_found",
        "image",
        [r(PROV, status="inconclusive", confidence=0.0), r(DEEPFAKE, confidence=0.6)],
        "could_not_verify",
        "unknown",
        None,
        50,
    ),
    Case(
        "provenance_error",
        "image",
        [r(PROV, status="error", confidence=0.0), r(DEEPFAKE)],
        "could_not_verify",
        "unknown",
        unavailable=[PROV],
    ),
    Case(
        "fact_check_error_on_clean_text",
        "text",
        [r(SCAM, confidence=0.9), r(FACT, status="error", confidence=0.0)],
        "no_issues",
        "low",
        SCAM,
        7,
        unavailable=[FACT],
    ),
    Case("ordinary_text", "text", [r(SCAM, confidence=0.9)], "no_issues", "low", SCAM, 7),
    Case(
        "weak_signal_blocks_no_issues",
        "text",
        [r(SCAM, confidence=0.9), r(FACT, "debunked", 0.3)],
        "could_not_verify",
        "unknown",
    ),
    Case("empty_results", "text", [], "could_not_verify", "unknown", None, 50),
    Case(
        "source_confirmed",
        "image",
        [r(PROV, origin={"credible": True}), r(CAPTION)],
        "no_issues",
        "low",
        PROV,
        source_confirmed=True,
    ),
    Case(
        "source_not_confirmed_with_listed_mismatches",
        "image",
        [r(PROV, origin={"credible": True}), r(CAPTION, mismatches=[{"field": "event"}])],
        "no_issues",
        "low",
        PROV,
    ),
    Case(
        "source_confirmed_with_empty_mismatch_list",
        "image",
        [r(PROV, origin={"credible": True}), r(CAPTION, mismatches=[])],
        "no_issues",
        "low",
        PROV,
        source_confirmed=True,
    ),
    Case(
        "source_not_confirmed_when_caption_inconclusive",
        "image",
        [r(PROV, origin={"credible": True}), r(CAPTION, status="inconclusive", confidence=0.0)],
        "no_issues",
        "low",
        PROV,
    ),
    Case(
        "source_not_confirmed_without_caption",
        "image",
        [r(PROV, origin={"credible": True})],
        "no_issues",
        "low",
        PROV,
    ),
    Case(
        "source_not_confirmed_when_not_credible",
        "image",
        [r(PROV, origin={"credible": "yes"}), r(CAPTION)],
        "no_issues",
        "low",
        PROV,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_rules(app_config: AppConfig, case: Case) -> None:
    verdict = decide(case.results, case.kind, app_config.verdict_rules)

    assert (verdict.label, verdict.band) == (case.label, case.band)
    assert verdict.deciding_checker == case.deciding
    if case.score is not None:
        assert verdict.score == case.score
    assert verdict.unavailable == case.unavailable
    assert verdict.source_confirmed is case.source_confirmed
    assert verdict.contributing == [(x.checker, x.status, x.signal) for x in case.results]


def test_highest_confidence_clause_decides(app_config: AppConfig) -> None:
    results = [r(PROV, "old_media", 0.65), r(CAPTION, "out_of_context", 0.9)]

    verdict = decide(results, "image", app_config.verdict_rules)

    assert verdict.deciding_checker == CAPTION
    assert verdict.score == 66


@pytest.mark.parametrize(
    ("band", "confidence", "score"),
    [
        ("high", 0.0, 70),
        ("high", 1.0, 100),
        ("low", 1.0, 5),
        ("low", 0.0, 25),
        ("unknown", 0.3, 50),
    ],
)
def test_band_score(app_config: AppConfig, band: Any, confidence: float, score: int) -> None:
    assert band_score(band, confidence, app_config.verdict_rules) == score


def test_shipped_rules_are_in_priority_order(app_config: AppConfig) -> None:
    rules: VerdictRulesConfig = app_config.verdict_rules

    assert [rule.id for rule in rules.rules] == [
        "scam",
        "old_or_out_of_context",
        "debunked",
        "synthetic",
        "no_issues",
        "fallback",
    ]
    assert rules.no_issues_requires_no_signals is True
