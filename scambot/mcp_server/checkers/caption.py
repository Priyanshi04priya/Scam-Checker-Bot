"""Comparing what a message says about a picture with where the picture first appeared.

Dates are compared in code. The model only looks for differences in place, people and event, and
every value it quotes must appear in the texts it was given, so it cannot invent a mismatch.
"""

import asyncio
import datetime as dt
import json
from typing import Any, Final, Literal

from pydantic import BaseModel

from scambot.core.config import CaptionThresholds, ProvenanceThresholds
from scambot.core.contracts import CheckerResult, CheckerStatus, Evidence, Signal
from scambot.core.llm import LLMClient, LLMError, LLMUnavailableError, usage_scope
from scambot.core.logging import get_logger

CHECKER: Final = "check_caption_context"
MismatchField = Literal["date", "place", "people", "event"]

_log = get_logger(component="caption")


class Mismatch(BaseModel):
    field: MismatchField
    claimed: str
    original: str


class CaptionReply(BaseModel):
    """LLM output schema; the values are checked in code."""

    mismatches: list[Mismatch]


def _parse_date(value: object) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


class CaptionChecker:
    def __init__(
        self, llm: LLMClient, thresholds: CaptionThresholds, provenance: ProvenanceThresholds
    ) -> None:
        self._llm = llm
        self._limits = thresholds
        self._old_days = provenance.old_media_days

    @staticmethod
    def _result(  # noqa: PLR0913
        status: CheckerStatus,
        details: dict[str, Any],
        *,
        signal: Signal = "none",
        confidence: float = 0.0,
        evidence: list[Evidence] | None = None,
        error: str | None = None,
    ) -> CheckerResult:
        return CheckerResult(
            checker=CHECKER,
            status=status,
            signal=signal,
            confidence=confidence,
            evidence=evidence or [],
            duration_ms=0,
            error=error,
            details=details,
        )

    def _validated(
        self, mismatches: list[Mismatch], origin_text: str, claim_text: str, place: str | None
    ) -> list[Mismatch]:
        limit = self._limits.max_field_chars
        kept: list[Mismatch] = []
        seen: set[str] = set()
        origin_low, claim_low = origin_text.lower(), claim_text.lower()
        for item in mismatches:
            claimed, original = item.claimed.strip(), item.original.strip()
            if item.field == "date" or item.field in seen or not claimed or not original:
                continue
            if original.lower() not in origin_low:
                continue
            if claimed.lower() not in claim_low and claimed.lower() != (place or "").lower():
                continue
            seen.add(item.field)
            kept.append(
                Mismatch(field=item.field, claimed=claimed[:limit], original=original[:limit])
            )
        return kept[: self._limits.max_mismatches]

    async def _ask(
        self, variables: dict[str, str]
    ) -> tuple[list[Mismatch], str | None, list[dict[str, Any]]]:
        """Mismatches from the model, a failure reason (or None) and the usage records."""
        with usage_scope() as usages:
            try:
                reply = await asyncio.wait_for(
                    self._llm.structured("caption_compare", variables, CaptionReply),
                    self._limits.llm_timeout_s,
                )
            except LLMUnavailableError:
                return [], "llm_unavailable", []
            except (LLMError, TimeoutError) as exc:
                _log.info("caption_compare_failed", error=type(exc).__name__)
                return [], "llm_failed", [u.model_dump(mode="json") for u in usages]
        return reply.mismatches, None, [u.model_dump(mode="json") for u in usages]

    async def check(
        self,
        caption: str,
        claimed_date: dt.date | None,
        claimed_place: str | None,
        origin: dict[str, Any] | None,
        received_on: dt.date | None,
    ) -> CheckerResult:
        """Compare the message's claim with ``origin``, the first credible page for the picture."""
        origin_date = _parse_date(origin.get("date")) if isinstance(origin, dict) else None
        if not isinstance(origin, dict) or origin_date is None or not origin.get("url"):
            return self._result("inconclusive", {"reason": "no_origin"})
        text = caption.strip()[: self._limits.max_text_chars]
        place = (claimed_place or "").strip() or None
        if not text and claimed_date is None and place is None:
            return self._result("inconclusive", {"reason": "no_claim"})
        publisher = str(origin.get("publisher") or "")
        title = str(origin.get("title") or "")
        description = str(origin.get("description") or "")
        found: list[Mismatch] = []
        if claimed_date is not None and abs((claimed_date - origin_date).days) > self._old_days:
            found.append(
                Mismatch(
                    field="date",
                    claimed=claimed_date.isoformat(),
                    original=origin_date.isoformat(),
                )
            )
        variables = {
            "received_on": received_on.isoformat() if received_on else "",
            "caption": text,
            "claimed_date": claimed_date.isoformat() if claimed_date else "",
            "claimed_place": place or "",
            "origin_json": json.dumps(
                {
                    "publisher": publisher,
                    "title": title,
                    "description": description,
                    "date": origin_date.isoformat(),
                    "domain": origin.get("domain") or "",
                },
                ensure_ascii=False,
            ),
        }
        asked, failure, usage = await self._ask(variables)
        found += self._validated(
            asked, f"{title} {description} {publisher}", f"{text} {place or ''}", place
        )
        details: dict[str, Any] = {
            "mismatches": [m.model_dump() for m in found],
            "origin_url": str(origin["url"]),
            "llm_usage": usage,
        }
        if failure is not None:
            details["reason"] = failure
        return self._finish(found, failure, details, origin, origin_date, publisher)

    def _finish(  # noqa: PLR0913, PLR0917
        self,
        found: list[Mismatch],
        failure: str | None,
        details: dict[str, Any],
        origin: dict[str, Any],
        origin_date: dt.date,
        publisher: str,
    ) -> CheckerResult:
        confirmed = {m.field for m in found if m.field in ("date", "place")}
        if not confirmed and failure is not None:
            if failure == "llm_unavailable":
                return self._result("inconclusive", details)
            return self._result("error", details, error=failure)
        evidence = [
            Evidence(
                summary=f"Original source: {publisher or origin.get('domain') or 'news page'}",
                url=str(origin["url"]),
                source=publisher or None,
                date=origin_date,
            )
        ]
        evidence += [
            Evidence(summary=f"{m.field}: claimed '{m.claimed}', original '{m.original}'")
            for m in found
        ]
        if confirmed:
            both = len(confirmed) == 2
            confidence = (
                self._limits.confidence_date_and_place if both else self._limits.confidence_single
            )
            return self._result(
                "ok", details, signal="out_of_context", confidence=confidence, evidence=evidence
            )
        return self._result(
            "ok",
            details,
            confidence=self._limits.confidence_single,
            evidence=evidence,
        )
