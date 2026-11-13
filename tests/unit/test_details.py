"""The Details and History messages."""

import datetime as dt
import re

import pytest

from scambot.agent.details import render_details, render_history
from scambot.core.config import AppConfig
from scambot.core.contracts import CheckerResult, Evidence
from scambot.core.locales import Locales
from scambot.core.store import DetailsData, HistoryItem

CHECKERS = [
    "analyze_scam_patterns",
    "fact_check_claim",
    "check_provenance",
    "check_caption_context",
    "detect_deepfake",
    "detect_voice_clone",
]
OUTCOMES = {
    "found": ("ok", "scam"),
    "clear": ("ok", "none"),
    "unsure": ("inconclusive", "none"),
    "unavailable": ("error", "none"),
}


def result(checker: str, outcome: str, urls: tuple[str, ...] = ()) -> CheckerResult:
    status, signal = OUTCOMES[outcome]
    return CheckerResult(
        checker=checker,
        status=status,  # type: ignore[arg-type]
        signal=signal,  # type: ignore[arg-type]
        confidence=0.77,
        duration_ms=5,
        evidence=[Evidence(summary="page text that must not appear 12345", url=u) for u in urls],
        details={"signals": ["urgency", "otp_request"]},
    )


@pytest.mark.parametrize("lang", ["en", "hi"])
@pytest.mark.parametrize("outcome", list(OUTCOMES))
def test_every_outcome_of_every_checker(
    locales: Locales, app_config: AppConfig, lang: str, outcome: str
) -> None:
    texts = locales.get(lang)
    data = DetailsData(
        "likely_scam", "high", 94, lang, [result(name, outcome) for name in CHECKERS]
    )

    text = render_details(data, texts)

    for name in CHECKERS:
        assert getattr(getattr(texts.details.checkers, name), outcome) in text
    assert texts.disclaimer in text
    assert "12345" not in text
    assert not re.search(r"\d", text.replace("🔴", ""))  # no scores, counts or ids
    lowered = text.casefold()
    assert not any(
        term in lowered
        for term in app_config.thresholds.reply.banned_terms
        if term.isascii() and len(term) > 3
    )


def test_scam_signals_are_listed_in_plain_words(locales: Locales) -> None:
    texts = locales.get("en")
    data = DetailsData("likely_scam", "high", 90, "en", [result(CHECKERS[0], "found")])

    text = render_details(data, texts)

    assert texts.verdict.signals.urgency in text
    assert texts.verdict.signals.otp_request in text


def test_at_most_three_sources_are_shown(locales: Locales) -> None:
    urls = tuple(f"https://example.org/{i}" for i in range(6))
    data = DetailsData("false_misleading", "high", 90, "en", [result(CHECKERS[1], "found", urls)])

    text = render_details(data, locales.get("en"))

    assert text.count("https://example.org/") == 3


def test_expired_when_nothing_is_left(locales: Locales) -> None:
    data = DetailsData("no_issues", "low", 5, "en", [])

    assert render_details(data, locales.get("en")) == locales.get("en").details.expired


def test_history_lists_newest_first_without_text(locales: Locales) -> None:
    texts = locales.get("en")
    when = dt.datetime(2026, 11, 13, 20, 0, tzinfo=dt.UTC)  # 01:30 the next day in India
    items = [
        HistoryItem(when, "replied", "likely_scam", "high"),
        HistoryItem(when, "failed", None, None),
    ]

    lines = render_history(items, texts).splitlines()

    assert lines[0] == texts.history.title
    assert lines[1].startswith("1. 14-11-2026 01:30 · ")
    assert texts.verdict.labels.likely_scam in lines[1]
    assert lines[2].endswith(texts.history.no_result)
    assert render_history([], texts) == texts.history.empty
