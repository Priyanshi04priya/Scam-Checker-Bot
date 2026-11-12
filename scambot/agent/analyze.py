"""Request analysis: the claims in a message, its language and any extra checks it needs."""

import asyncio
import datetime as dt
import re
from typing import Final

from pydantic import BaseModel

from scambot.core.contracts import CheckerName, Preprocessed
from scambot.core.llm import LLMClient, LLMError
from scambot.core.logging import get_logger

MAX_CLAIMS: Final = 3
MAX_CLAIM_CHARS: Final = 300
MAX_PLACE_CHARS: Final = 100
_LANGUAGE_RE: Final = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{4})?$")

_log = get_logger(component="analyze")


class Analysis(BaseModel):
    """Also the LLM output schema, so it has no numeric or length constraints."""

    claims: list[str]
    reply_language: str
    claimed_date: dt.date | None
    claimed_place: str | None
    extra_checkers: list[CheckerName]
    reason: str


def _normalise(analysis: Analysis) -> Analysis:
    claims = [claim.strip()[:MAX_CLAIM_CHARS] for claim in analysis.claims if claim.strip()]
    language = analysis.reply_language.strip()
    place = analysis.claimed_place.strip()[:MAX_PLACE_CHARS] if analysis.claimed_place else None
    return analysis.model_copy(
        update={
            "claims": claims[:MAX_CLAIMS],
            "reply_language": language if _LANGUAGE_RE.match(language) else "",
            "claimed_place": place or None,
            "extra_checkers": list(dict.fromkeys(analysis.extra_checkers)),
        }
    )


async def analyze(
    pre: Preprocessed, *, llm: LLMClient, received_on: dt.date, timeout_s: float
) -> Analysis | None:
    """Analyse the message with the LLM; None when there is no text or the call fails."""
    if not pre.text.strip():
        return None
    variables = {"content": pre.text, "kind": pre.kind, "received_on": received_on.isoformat()}
    try:
        analysis = await asyncio.wait_for(llm.structured("analyze", variables, Analysis), timeout_s)
    except (LLMError, TimeoutError) as exc:
        _log.info("analysis_failed", error=type(exc).__name__)
        return None
    return _normalise(analysis)
