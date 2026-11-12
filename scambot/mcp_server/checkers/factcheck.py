"""Matching claims against published fact-checks.

Each claim is searched in the Fact Check Tools API and in the local index (nearest embeddings
and full text); the fused candidates are judged in one LLM call. Only a fact-check of the same
claim that rates it false or misleading makes the claim ``debunked``.
"""

import asyncio
import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel

from scambot.core.config import FactCheckThresholds, SourcesConfig
from scambot.core.contracts import CheckerResult, CheckerStatus, Evidence, RatingPolarity, Signal
from scambot.core.llm import LLMClient, LLMError, LLMUnavailableError, usage_scope
from scambot.core.logging import get_logger
from scambot.factcheck_index.embed import Embedder
from scambot.factcheck_index.ratings import rating_polarity
from scambot.factcheck_index.records import (
    MAX_CLAIM_CHARS,
    MAX_RATING_CHARS,
    MAX_TITLE_CHARS,
    ArticleRecord,
    ClaimReview,
    record_from_review,
)
from scambot.factcheck_index.search import FactcheckIndex, Retrieval, fuse, retrieve
from scambot.mcp_server.clients.factcheck_tools import FactCheckToolsError

CHECKER: Final = "fact_check_claim"
DEBUNKING: Final = frozenset({"false", "misleading"})
CONFIRMING: Final = frozenset({"true", "mixed"})

StepStatus = Literal["ok", "error", "skipped"]
RelevanceStatus = Literal["ok", "error", "unavailable", "skipped"]

_log = get_logger(component="factcheck")


class ClaimSearch(Protocol):
    async def search(self, query: str, *, page_size: int) -> list[ClaimReview]: ...


class CandidateJudgment(BaseModel):
    candidate: int
    relation: Literal["same", "related", "different"]
    polarity: RatingPolarity


class RelevanceReply(BaseModel):
    """LLM output schema; numbers are checked in code."""

    judgments: list[CandidateJudgment]


@dataclass(frozen=True)
class FactCheckSources:
    api: ClaimSearch | None
    index: FactcheckIndex | None
    embedder: Embedder | None


@dataclass(frozen=True)
class _Match:
    claim: int
    record: ArticleRecord
    relation: Literal["same", "related"]
    polarity: RatingPolarity
    polarity_from: Literal["rating", "llm"]


@dataclass
class _ClaimOutcome:
    api: StepStatus = "skipped"
    index: StepStatus = "skipped"
    relevance: RelevanceStatus = "skipped"
    candidates: int = 0
    matches: list[_Match] = field(default_factory=list)

    @property
    def searched(self) -> bool:
        return "ok" in (self.api, self.index)

    @property
    def resolved(self) -> bool:
        return self.searched and (self.candidates == 0 or self.relevance == "ok")


def _overall[S: str](statuses: Sequence[S], order: Sequence[S]) -> S:
    """The first status in ``order`` that any claim has (the last one when none matches)."""
    return next((status for status in order if status in statuses), order[-1])


class FactCheckChecker:
    def __init__(
        self,
        sources: FactCheckSources,
        llm: LLMClient,
        thresholds: FactCheckThresholds,
        catalogue: SourcesConfig,
    ) -> None:
        self._sources = sources
        self._llm = llm
        self._thresholds = thresholds
        self._catalogue = catalogue
        self._use_index = sources.index is not None and sources.embedder is not None

    def _claims(self, claims: Sequence[str]) -> list[str]:
        limit = self._thresholds.max_claim_chars
        cleaned = [claim.strip()[:limit] for claim in claims if claim.strip()]
        return list(dict.fromkeys(cleaned))[: self._thresholds.max_claims]

    def _name(self, source: str) -> str:
        known = self._catalogue.by_id(source)
        return known.name if known else source

    # --- search ----------------------------------------------------------------------------

    async def _api_search(self, claim: str) -> tuple[StepStatus, list[ArticleRecord]]:
        api = self._sources.api
        if api is None:
            return "skipped", []
        try:
            reviews = await asyncio.wait_for(
                api.search(claim, page_size=self._thresholds.api_page_size),
                self._thresholds.api_timeout_s,
            )
        except (FactCheckToolsError, TimeoutError) as exc:
            _log.info("factcheck_api_failed", error=type(exc).__name__)
            return "error", []
        records = [record_from_review(review, self._catalogue) for review in reviews]
        return "ok", [record for record in records if record is not None]

    async def _index_search(self, claim: str) -> tuple[StepStatus, Retrieval]:
        empty = Retrieval(vector=[], text=[])
        index, embedder = self._sources.index, self._sources.embedder
        if index is None or embedder is None:
            return "skipped", empty
        try:
            found = await asyncio.wait_for(
                retrieve(index, embedder, claim, self._thresholds.top_k),
                self._thresholds.index_timeout_s,
            )
        except Exception as exc:  # a broken index must not stop the API answer
            _log.info("factcheck_index_failed", error=type(exc).__name__)
            return "error", empty
        return "ok", found

    # --- relevance -------------------------------------------------------------------------

    def _candidates_json(self, candidates: Sequence[ArticleRecord]) -> str:
        items = [
            {
                "candidate": number,
                "source": self._name(record.source),
                "title": record.title[:MAX_TITLE_CHARS],
                "claim_checked": record.claim[:MAX_CLAIM_CHARS],
                "rating": record.rating[:MAX_RATING_CHARS] if record.rating else None,
                "date": record.published_at.date().isoformat() if record.published_at else None,
            }
            for number, record in enumerate(candidates, start=1)
        ]
        return json.dumps(items, ensure_ascii=False)

    async def _judge(
        self, claim: str, candidates: Sequence[ArticleRecord]
    ) -> tuple[RelevanceStatus, list[CandidateJudgment]]:
        variables = {"claim": claim, "candidates_json": self._candidates_json(candidates)}
        try:
            reply = await asyncio.wait_for(
                self._llm.structured("factcheck_relevance", variables, RelevanceReply),
                self._thresholds.relevance_timeout_s,
            )
        except LLMUnavailableError:
            return "unavailable", []
        except (LLMError, TimeoutError) as exc:
            _log.info("factcheck_relevance_failed", error=type(exc).__name__)
            return "error", []
        return "ok", reply.judgments

    def _matches(
        self,
        claim_number: int,
        candidates: Sequence[ArticleRecord],
        judgments: Sequence[CandidateJudgment],
    ) -> list[_Match]:
        matches: list[_Match] = []
        seen: set[int] = set()
        for judgment in judgments:
            number = judgment.candidate
            if not 1 <= number <= len(candidates) or number in seen:
                continue
            seen.add(number)
            if judgment.relation != "same":
                continue
            record = candidates[number - 1]
            rated = rating_polarity(record.rating, self._catalogue.ratings)
            if rated != "unrated":
                matches.append(_Match(claim_number, record, "same", rated, "rating"))
            else:
                matches.append(_Match(claim_number, record, "same", judgment.polarity, "llm"))
        return matches

    async def _check_claim(self, number: int, claim: str) -> _ClaimOutcome:
        (api_status, api_hits), (index_status, found) = await asyncio.gather(
            self._api_search(claim), self._index_search(claim)
        )
        outcome = _ClaimOutcome(api=api_status, index=index_status)
        candidates = fuse([api_hits, found.vector, found.text], self._thresholds.rrf_k)[
            : self._thresholds.max_candidates
        ]
        outcome.candidates = len(candidates)
        if candidates:
            outcome.relevance, judgments = await self._judge(claim, candidates)
            outcome.matches = self._matches(number, candidates, judgments)
        return outcome

    # --- result ----------------------------------------------------------------------------

    def _confidence(self, match: _Match) -> float:
        if match.polarity_from == "rating":
            return self._thresholds.confidence_rated
        return self._thresholds.confidence_llm_polarity

    def _evidence(self, matches: Sequence[_Match]) -> list[Evidence]:
        oldest = dt.datetime.min.replace(tzinfo=dt.UTC)
        ordered = sorted(
            matches,
            key=lambda m: (
                m.polarity not in DEBUNKING,
                -(m.record.published_at or oldest).timestamp(),
            ),
        )
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for match in ordered:
            record = match.record
            if record.url in seen:
                continue
            seen.add(record.url)
            name = self._name(record.source)
            evidence.append(
                Evidence(
                    summary=f"{name}: {record.title} ({record.rating or match.polarity})",
                    url=record.url,
                    source=name,
                    date=record.published_at.date() if record.published_at else None,
                )
            )
        return evidence[: self._thresholds.max_evidence]

    @staticmethod
    def _decide(outcomes: Sequence[_ClaimOutcome]) -> tuple[CheckerStatus, str | None]:
        if not outcomes:
            return "inconclusive", None
        if all(outcome.resolved for outcome in outcomes):
            return "ok", None
        if any(o.candidates and o.relevance == "unavailable" for o in outcomes):
            return "inconclusive", None
        if any(not outcome.searched for outcome in outcomes):
            return "error", "search_failed"
        return "error", "relevance_failed"

    def _result(
        self, outcomes: Sequence[_ClaimOutcome], usage: list[dict[str, Any]]
    ) -> CheckerResult:
        matches = [match for outcome in outcomes for match in outcome.matches]
        debunked = [match for match in matches if match.polarity in DEBUNKING]
        status, error = self._decide(outcomes)
        signal: Signal = "none"
        confidence = 0.0
        if debunked:
            status, error, signal = "ok", None, "debunked"
            confidence = max(self._confidence(match) for match in debunked)
        elif status == "ok" and any(match.polarity in CONFIRMING for match in matches):
            confidence = self._thresholds.confidence_rated
        return CheckerResult(
            checker=CHECKER,
            status=status,
            signal=signal,
            confidence=confidence,
            evidence=self._evidence(matches) if status == "ok" else [],
            duration_ms=0,
            error=error,
            details={
                "claims": len(outcomes),
                "api": _overall([o.api for o in outcomes], ("error", "ok", "skipped")),
                "index": _overall([o.index for o in outcomes], ("error", "ok", "skipped")),
                "relevance": _overall(
                    [o.relevance for o in outcomes], ("error", "unavailable", "ok", "skipped")
                ),
                "candidates": sum(outcome.candidates for outcome in outcomes),
                "matches": [self._match_details(match) for match in matches],
                "llm_usage": usage,
            },
        )

    def _match_details(self, match: _Match) -> dict[str, Any]:
        record = match.record
        return {
            "claim": match.claim,
            "url": record.url,
            "source": record.source,
            "title": record.title,
            "rating": record.rating,
            "polarity": match.polarity,
            "polarity_from": match.polarity_from,
            "relation": match.relation,
            "date": record.published_at.date().isoformat() if record.published_at else None,
        }

    async def check(self, claims: Sequence[str]) -> CheckerResult:
        """Look up each claim; ``duration_ms`` is filled in by the tool wrapper."""
        if self._sources.api is None and not self._use_index:
            return CheckerResult(
                checker=CHECKER,
                status="error",
                signal="none",
                confidence=0.0,
                duration_ms=0,
                error="no_sources",
            )
        cleaned = self._claims(claims)
        with usage_scope() as usages:
            outcomes = await asyncio.gather(
                *(self._check_claim(number, claim) for number, claim in enumerate(cleaned, 1))
            )
        return self._result(outcomes, [usage.model_dump(mode="json") for usage in usages])
