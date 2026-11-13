"""The Details and History messages, built from stored results only (no LLM call)."""

from collections.abc import Sequence
from typing import Final

from scambot.agent.reply import BAND_EMOJI, headline, signal_names
from scambot.core.contracts import Band, CheckerResult, Verdict
from scambot.core.locales import IST, LocaleTexts
from scambot.core.store import DetailsData, HistoryItem

_CHECKER_ORDER: Final = (
    "analyze_scam_patterns",
    "fact_check_claim",
    "check_provenance",
    "check_caption_context",
    "detect_deepfake",
    "detect_voice_clone",
)
_ICONS: Final = {"found": "⚠️", "clear": "✅", "unsure": "❔", "unavailable": "⛔"}


def _outcome(result: CheckerResult) -> str:
    if result.status == "error":
        return "unavailable"
    if result.status == "inconclusive":
        return "unsure"
    return "found" if result.signal != "none" else "clear"


def _sources(results: Sequence[CheckerResult], limit: int) -> list[str]:
    urls: dict[str, None] = {}
    for result in results:
        for item in result.evidence:
            if item.url and item.url.lower().startswith(("http://", "https://")):
                urls.setdefault(item.url, None)
    return list(urls)[:limit]


def render_details(data: DetailsData, texts: LocaleTexts, max_sources: int = 3) -> str:
    """One line per check with what it found, then the sources and the disclaimer."""
    if not data.results:
        return texts.details.expired
    verdict = Verdict.model_validate({"label": data.label, "band": data.band, "score": data.score})
    ordered = sorted(
        data.results,
        key=lambda r: _CHECKER_ORDER.index(r.checker) if r.checker in _CHECKER_ORDER else 99,
    )
    lines: list[str] = []
    for result in ordered:
        names = getattr(texts.details.checkers, result.checker, None)
        if names is None:
            continue
        outcome = _outcome(result)
        line = f"{_ICONS[outcome]} {getattr(texts.verdict.checkers, result.checker)}: "
        line += getattr(names, outcome)
        if result.checker == "analyze_scam_patterns" and outcome == "found":
            signals = [getattr(texts.verdict.signals, name) for name in signal_names(result)]
            if signals:
                line += f" ({', '.join(signals)})"
        lines.append(line)
    blocks = [texts.details.title, headline(verdict, texts), "\n".join(lines)]
    sources = _sources(ordered, max_sources)
    if sources:
        blocks.append(f"{texts.verdict.sources}:\n" + "\n".join(f"• {url}" for url in sources))
    blocks.append(texts.disclaimer)
    return "\n\n".join(block for block in blocks if block)


def render_history(items: Sequence[HistoryItem], texts: LocaleTexts) -> str:
    """The user's last checks, newest first: date, risk level and result. Never message text."""
    if not items:
        return texts.history.empty
    lines = [texts.history.title]
    for number, item in enumerate(items, start=1):
        when = item.created_at.astimezone(IST).strftime("%d-%m-%Y %H:%M")
        if item.label is None or item.band is None:
            result = texts.history.no_result
        else:
            band: Band = item.band  # type: ignore[assignment]
            result = (
                f"{BAND_EMOJI.get(band, '')} {getattr(texts.verdict.bands, item.band, item.band)}"
                f" · {getattr(texts.verdict.labels, item.label, item.label)}"
            )
        lines.append(f"{number}. {when} · {result}")
    return "\n".join(lines)
