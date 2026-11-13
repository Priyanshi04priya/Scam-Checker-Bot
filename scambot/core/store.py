"""Persistence of requests, checker runs and verdicts."""

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.contracts import ExecutedCheck, Verdict
from scambot.core.logging import get_logger
from scambot.core.models import CheckerRun, Request, User, VerdictRow

RequestStatus = Literal["received", "replied", "failed", "timeout"]

_log = get_logger(component="store")


@dataclass(frozen=True)
class NewRequest:
    job_id: str
    user_hash: str
    channel: str
    kind: str
    sha256: str | None
    language: str | None
    caption_hash: str | None = None


@dataclass(frozen=True)
class StartedRequest:
    id: uuid.UUID
    status: RequestStatus
    created: bool


@dataclass(frozen=True)
class StoredResult:
    checks: Sequence[ExecutedCheck]
    verdict: Verdict | None
    reply_text: str
    reply_language: str
    claims: list[str]
    language: str | None
    cost_usd: Decimal
    sha256: str | None = None
    phash: int | None = None


class RequestStore(Protocol):
    async def start(self, request: NewRequest) -> StartedRequest:
        """Create the request row, or return the existing one for the same job."""
        ...

    async def save(self, request_id: uuid.UUID, result: StoredResult) -> None:
        """Store the checker runs and verdict, replacing any from an earlier attempt."""
        ...

    async def set_status(
        self, request_id: uuid.UUID, status: RequestStatus, *, at: dt.datetime | None = None
    ) -> None:
        """Change the status; ``at`` is the reply time when the status is ``replied``."""
        ...


def _started(row: Request, *, created: bool) -> StartedRequest:
    status: RequestStatus = row.status  # type: ignore[assignment]
    return StartedRequest(id=row.id, status=status, created=created)


class SqlRequestStore:
    """``RequestStore`` on SQLAlchemy; portable between PostgreSQL and SQLite."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def _existing(self, job_id: str) -> Request | None:
        async with self._sessions() as session:
            result = await session.execute(select(Request).where(Request.job_id == job_id))
            return result.scalar_one_or_none()

    async def _insert(self, request: NewRequest) -> Request:
        async with self._sessions() as session, session.begin():
            if await session.get(User, request.user_hash) is None:
                session.add(User(user_hash=request.user_hash))
                await session.flush()
            row = Request(
                job_id=request.job_id,
                user_hash=request.user_hash,
                channel=request.channel,
                kind=request.kind,
                sha256=request.sha256,
                caption_hash=request.caption_hash,
                language=request.language,
                claims=[],
                status="received",
            )
            session.add(row)
        return row

    async def start(self, request: NewRequest) -> StartedRequest:
        for _attempt in range(2):
            existing = await self._existing(request.job_id)
            if existing is not None:
                return _started(existing, created=False)
            try:
                row = await self._insert(request)
            except IntegrityError:
                # A concurrent attempt inserted the same job or user first: look again.
                continue
            _log.info("request_started", request_id=str(row.id))
            return _started(row, created=True)
        existing = await self._existing(request.job_id)
        if existing is None:
            msg = "request could not be created"
            raise RuntimeError(msg)
        return _started(existing, created=False)

    async def save(self, request_id: uuid.UUID, result: StoredResult) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(CheckerRun).where(CheckerRun.request_id == request_id))
            await session.execute(delete(VerdictRow).where(VerdictRow.request_id == request_id))
            session.add_all(
                CheckerRun(
                    request_id=request_id,
                    checker=check.result.checker,
                    input_hash=check.input_hash,
                    status=check.result.status,
                    signal=check.result.signal,
                    confidence=check.result.confidence,
                    evidence=[item.model_dump(mode="json") for item in check.result.evidence],
                    details=check.result.model_dump(mode="json")["details"],
                    duration_ms=check.result.duration_ms,
                    error=check.result.error,
                )
                for check in result.checks
            )
            if result.verdict is not None:
                session.add(
                    VerdictRow(
                        request_id=request_id,
                        label=result.verdict.label,
                        band=result.verdict.band,
                        score=result.verdict.score,
                        deciding_checker=result.verdict.deciding_checker,
                        reply_text=result.reply_text,
                        reply_language=result.reply_language,
                    )
                )
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    claims=result.claims,
                    # Keep the language recorded at start when analysis found none.
                    language=func.coalesce(result.language, Request.language),
                    cost_usd=result.cost_usd,
                    sha256=func.coalesce(result.sha256, Request.sha256),
                    phash=func.coalesce(result.phash, Request.phash),
                )
            )
        _log.info("request_saved", request_id=str(request_id), checks=len(result.checks))

    async def set_status(
        self, request_id: uuid.UUID, status: RequestStatus, *, at: dt.datetime | None = None
    ) -> None:
        values: dict[str, object] = {"status": status}
        if status == "replied":
            values["replied_at"] = at or dt.datetime.now(dt.UTC)
        async with self._sessions() as session, session.begin():
            await session.execute(update(Request).where(Request.id == request_id).values(values))
