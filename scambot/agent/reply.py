"""The reply message text: headline, explanation, notes, sources and disclaimer."""

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

from pydantic import BaseModel

from scambot.agent.analyze import Analysis
from scambot.core.config import ReplyConfig
from scambot.core.contracts import Band, CheckerResult, ScamSignal, Verdict, VerdictLabel
from scambot.core.llm import LLMClient, LLMError
from scambot.core.locales import (
    CHECKS_PLACEHOLDER,
    HINGLISH,
    SIGNALS_PLACEHOLDER,
    Locales,
    LocaleTexts,
)
from scambot.core.logging import get_logger

BAND_EMOJI: Final[dict[Band, str]] = {"high": "🔴", "medium": "🟠", "low": "🟢", "unknown": "⚪"}

# Labels whose explanation must never call the content genuine unless its source was confirmed.
_CAUTIOUS_LABELS: Final[frozenset[VerdictLabel]] = frozenset(
    {"no_issues", "could_not_verify", "possibly_ai"}
)
_URL_RE: Final = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\."
    r"(?:com|in|org|net|co|ly|io|xyz|info|gov|app|link|me|top|site|online)\b"
)
_PHONE_RUN_RE: Final = re.compile(r"\d[\d\s-]{6,}\d")
# A genuine term is only an assertion when no negation precedes it in the same clause.
_NEGATION_WINDOW: Final = 4
_NEGATION_RE: Final = re.compile(
    r"(?i)(?:not|no|never|cannot|nothing|neither|nor|without|\w+n['\u2019]t|नहीं|नही|न|मत)"
)
_CLAUSE_BREAK_RE: Final = re.compile(r"(?i)[.,;:!?\u0964]|\b(?:but|and|however|yet)\b|लेकिन|और|पर\s")
# "No doubt it is genuine" and "कोई शक नहीं कि यह सच है" affirm the term; they are not negations.
_AFFIRMING_RE: Final = re.compile(
    r"(?i)\b(?:no|without(?:\s+a)?|beyond(?:\s+a)?)\s+(?:doubt|question)\b|(?:शक|संदेह)\s+(?:नहीं|नही)"
)
_SIGNAL_NAMES: Final = frozenset(name for alias in get_args(ScamSignal) for name in get_args(alias))

_log = get_logger(component="reply")


class Explanation(BaseModel):
    """LLM output schema for the explanation call."""

    explanation: str


@dataclass(frozen=True)
class ReplyInputs:
    verdict: Verdict
    results: Sequence[CheckerResult]
    language: str
    use_llm: bool
    style: str | None = None


@dataclass(frozen=True)
class Reply:
    text: str
    explanation_source: Literal["llm", "template"]


def choose_reply_language(
    analysis: Analysis | None,
    preferred_lang: str | None,
    locales: Locales,
    user_lang: str | None = None,
) -> str:
    """The language of the fixed texts: the user's own choice, then the message's language
    when it is a loaded locale, then the language set in the user's app, then English."""
    if user_lang:
        return locales.resolve(user_lang)
    if analysis is not None and analysis.reply_language in locales.languages:
        return analysis.reply_language
    return locales.resolve(preferred_lang)


def reply_style(analysis: Analysis | None, user_lang: str | None) -> str | None:
    """``hi-Latn`` when the explanation should be written as Hinglish, else None."""
    if user_lang:
        return HINGLISH if user_lang == HINGLISH else None
    if analysis is not None and analysis.reply_language.lower() == HINGLISH.lower():
        return HINGLISH
    return None


def _deciding(inputs: ReplyInputs) -> CheckerResult | None:
    name = inputs.verdict.deciding_checker
    return next((r for r in inputs.results if r.checker == name), None)


def signal_names(result: CheckerResult | None) -> list[str]:
    if result is None:
        return []
    signals = result.details.get("signals", [])
    if not isinstance(signals, list):
        return []
    return [s for s in dict.fromkeys(signals) if isinstance(s, str) and s in _SIGNAL_NAMES]


def headline(verdict: Verdict, texts: LocaleTexts) -> str:
    band = getattr(texts.verdict.bands, verdict.band)
    label = getattr(texts.verdict.labels, verdict.label)
    return f"{BAND_EMOJI[verdict.band]} {band} · {label}"


def _unavailable(verdict: Verdict, texts: LocaleTexts) -> str:
    if not verdict.unavailable:
        return ""
    names = [getattr(texts.verdict.checkers, name, name) for name in verdict.unavailable]
    return texts.verdict.unavailable.replace(CHECKS_PLACEHOLDER, ", ".join(names))


def _voice_flagged(results: Sequence[CheckerResult]) -> bool:
    return any(
        r.checker == "detect_voice_clone" and r.status == "ok" and r.signal == "likely_synthetic"
        for r in results
    )


def _template(inputs: ReplyInputs, texts: LocaleTexts) -> str:
    template: str = getattr(texts.verdict.explanations, inputs.verdict.label)
    if inputs.verdict.label != "likely_scam":
        return template
    signals = [getattr(texts.verdict.signals, s) for s in signal_names(_deciding(inputs))]
    listed = ", ".join(signals) if signals else texts.verdict.signals_generic
    return template.replace(SIGNALS_PLACEHOLDER, listed)


def _sources(inputs: ReplyInputs, max_sources: int) -> list[str]:
    deciding = _deciding(inputs)
    ordered = ([deciding] if deciding else []) + [r for r in inputs.results if r is not deciding]
    urls: dict[str, None] = {}
    for result in ordered:
        for item in result.evidence:
            if item.url and item.url.lower().startswith(("http://", "https://")):
                urls.setdefault(item.url, None)
    return list(urls)[:max_sources]


def _llm_variables(inputs: ReplyInputs) -> dict[str, str]:
    verdict = inputs.verdict
    verdict_json = {
        "label": verdict.label,
        "band": verdict.band,
        "deciding_checker": verdict.deciding_checker,
        "source_confirmed": verdict.source_confirmed,
        "unavailable": verdict.unavailable,
    }
    results_json: list[dict[str, Any]] = [
        {
            "checker": r.checker,
            "status": r.status,
            "signal": r.signal,
            "confidence": round(r.confidence, 2),
            "evidence": [item.summary for item in r.evidence],
            "signals": signal_names(r),
        }
        for r in inputs.results
    ]
    return {
        "language": inputs.style or inputs.language,
        "verdict_json": json.dumps(verdict_json, ensure_ascii=False),
        "results_json": json.dumps(results_json, ensure_ascii=False),
    }


def _contains_term(text: str, term: str) -> bool:
    if term.isascii():
        return re.search(rf"(?i)\b{re.escape(term)}\b", text) is not None
    return term.casefold() in text.casefold()


def _term_spans(text: str, term: str) -> list[int]:
    if term.isascii():
        return [m.start() for m in re.finditer(rf"(?i)\b{re.escape(term)}\b", text)]
    return [m.start() for m in re.finditer(re.escape(term.casefold()), text.casefold())]


def _is_negated(text: str, start: int) -> bool:
    """True when a negation word precedes the term within the same clause."""
    clause = _AFFIRMING_RE.sub(" ", _CLAUSE_BREAK_RE.split(text[:start])[-1])
    words = clause.split()[-_NEGATION_WINDOW:]
    return any(_NEGATION_RE.fullmatch(word.strip("\"'()")) for word in words)


def _asserts_term(text: str, term: str) -> bool:
    """True when the text uses the term outside a negation ("does not prove it is true")."""
    return any(not _is_negated(text, start) for start in _term_spans(text, term))


def _content_problem(explanation: str, inputs: ReplyInputs, config: ReplyConfig) -> str | None:
    if _URL_RE.search(explanation):
        return "url"
    if _PHONE_RUN_RE.search(explanation):
        return "phone_number"
    if any(_contains_term(explanation, term) for term in config.banned_terms):
        return "banned_term"
    if (
        not inputs.verdict.source_confirmed
        and inputs.verdict.label in _CAUTIOUS_LABELS
        and any(_asserts_term(explanation, term) for term in config.genuine_terms)
    ):
        return "genuine_term"
    return None


def _rejection(
    explanation: str, summary_chars: int, inputs: ReplyInputs, config: ReplyConfig
) -> str | None:
    """Name of the first post-check the explanation fails, or None when it passes."""
    if not explanation:
        return "empty"
    if summary_chars > config.summary_max_chars:
        return "too_long"
    return _content_problem(explanation, inputs, config)


@dataclass(frozen=True)
class _Parts:
    headline: str
    voice_note: str
    unavailable: str


async def _llm_explanation(
    inputs: ReplyInputs, parts: _Parts, *, llm: LLMClient, config: ReplyConfig, timeout_s: float
) -> str | None:
    try:
        reply = await asyncio.wait_for(
            llm.structured("explanation", _llm_variables(inputs), Explanation), timeout_s
        )
    except (LLMError, TimeoutError) as exc:
        _log.info("explanation_rejected", check="llm_failed", error=type(exc).__name__)
        return None
    explanation = reply.explanation.strip()
    # The fixed voice note is not part of the budget: it is always added, so counting it
    # would push most LLM explanations onto the template.
    summary = "\n\n".join(p for p in (parts.headline, explanation, parts.unavailable) if p)
    problem = _rejection(explanation, len(summary), inputs, config)
    if problem is not None:
        _log.info("explanation_rejected", check=problem)
        return None
    return explanation


async def compose_reply(
    inputs: ReplyInputs,
    *,
    llm: LLMClient,
    locales: Locales,
    config: ReplyConfig,
    timeout_s: float,
) -> Reply:
    """Build the reply text, using the LLM explanation only when it passes every check."""
    texts = locales.get(inputs.language)
    parts = _Parts(
        headline(inputs.verdict, texts),
        texts.verdict.voice_note if _voice_flagged(inputs.results) else "",
        _unavailable(inputs.verdict, texts),
    )
    explanation: str | None = None
    if inputs.use_llm:
        explanation = await _llm_explanation(
            inputs, parts, llm=llm, config=config, timeout_s=timeout_s
        )
    source: Literal["llm", "template"] = "llm" if explanation else "template"
    if explanation is None:
        explanation = _template(inputs, texts)
    sources = _sources(inputs, config.max_sources)
    sources_text = (
        f"{texts.verdict.sources}:\n" + "\n".join(f"• {url}" for url in sources) if sources else ""
    )
    blocks = (
        parts.headline,
        explanation,
        parts.voice_note,
        parts.unavailable,
        sources_text,
        texts.disclaimer,
    )
    return Reply(text="\n\n".join(b for b in blocks if b), explanation_source=source)
