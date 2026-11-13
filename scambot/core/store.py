"""Persistence of requests, checker runs and verdicts."""

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol

from sqlalchemy import BigInteger, bindparam, cast, delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.contracts import CheckerResult, ExecutedCheck, Verdict
from scambot.core.logging import get_logger
from scambot.core.models import CheckerRun, Feedback, Request, User, VerdictRow
from scambot.core.purge import delete_user_rows, purge_rows

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
    duration_ms: int | None = None


@dataclass(frozen=True)
class HistoryItem:
    """One line of a user's history: when, how it ended and the verdict if there is one."""

    created_at: dt.datetime
    status: str
    label: str | None
    band: str | None


@dataclass(frozen=True)
class DetailsData:
    """What the Details button shows: the verdict and the checks behind it."""

    label: str
    band: str
    score: int
    reply_language: str
    results: list[CheckerResult]


@dataclass(frozen=True)
class Fingerprint:
    """What identifies a piece of content: its kind, its hashes and the hash of its caption."""

    kind: str
    sha256: str | None
    phash: int | None
    caption_hash: str | None


@dataclass(frozen=True)
class CachedVerdict:
    """An earlier finished verdict that can be shown again."""

    request_id: uuid.UUID
    match: Literal["exact", "phash"]
    distance: int
    checked_at: dt.datetime
    label: str
    band: str
    score: int
    deciding_checker: str | None
    reply_text: str
    reply_language: str


MASK_64 = 2**64 - 1


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit hashes, signed or unsigned."""
    return ((a ^ b) & MASK_64).bit_count()


def as_utc(value: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; everything stored is UTC."""
    return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


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

    async def find_cached(  # noqa: PLR0913
        self,
        fp: Fingerprint,
        *,
        window: dt.timedelta,
        max_distance: int,
        candidates: int,
        now: dt.datetime,
        exclude: uuid.UUID,
    ) -> CachedVerdict | None:
        """The newest finished verdict for the same content, or None."""
        ...

    async def save_cached(
        self,
        request_id: uuid.UUID,
        cached: CachedVerdict,
        fp: Fingerprint,
        duration_ms: int | None = None,
    ) -> None:
        """Record that ``request_id`` was answered from ``cached`` at no cost."""
        ...

    async def load_results(self, request_id: uuid.UUID) -> list[CheckerResult]:
        """The checker results saved for a request."""
        ...

    async def delete_user(self, user_hash: str) -> int:
        """Delete everything stored for a user; returns the number of requests removed."""
        ...

    async def purge_older_than(self, cutoff: dt.datetime) -> int:
        """Delete requests created before ``cutoff``; returns how many."""
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
                    duration_ms=result.duration_ms,
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

    def _servable(self, fp: Fingerprint, since: dt.datetime, exclude: uuid.UUID) -> list[Any]:
        """Conditions for a row whose verdict can be shown again."""
        caption = (
            Request.caption_hash.is_(None)
            if fp.caption_hash is None
            else Request.caption_hash == fp.caption_hash
        )
        failed_check = exists().where(
            CheckerRun.request_id == Request.id, CheckerRun.status == "error"
        )
        has_verdict = exists().where(VerdictRow.request_id == Request.id)
        return [
            Request.status == "replied",
            Request.cache_of.is_(None),
            Request.kind == fp.kind,
            caption,
            Request.replied_at >= since,
            Request.id != exclude,
            has_verdict,
            ~failed_check,
        ]

    async def find_cached(  # noqa: PLR0913
        self,
        fp: Fingerprint,
        *,
        window: dt.timedelta,
        max_distance: int,
        candidates: int,
        now: dt.datetime,
        exclude: uuid.UUID,
    ) -> CachedVerdict | None:
        conditions = self._servable(fp, now - window, exclude)
        async with self._sessions() as session:
            if fp.sha256 is not None:
                query = select(Request.id, Request.replied_at).where(
                    Request.sha256 == fp.sha256, *conditions
                )
                row = (
                    await session.execute(query.order_by(Request.replied_at.desc()).limit(1))
                ).first()
                if row is not None:
                    return await self._cached(session, row[0], row[1], "exact", 0)
            if fp.kind != "image" or fp.phash is None:
                return None
            found = await self._nearest(session, fp.phash, conditions, max_distance, candidates)
            if found is None:
                return None
            return await self._cached(session, found[0], found[1], "phash", found[2])

    async def _nearest(
        self,
        session: AsyncSession,
        phash: int,
        conditions: list[Any],
        max_distance: int,
        candidates: int,
    ) -> tuple[uuid.UUID, dt.datetime | None, int] | None:
        base = select(Request.id, Request.replied_at, Request.phash).where(
            Request.phash.is_not(None), *conditions
        )
        if session.get_bind().dialect.name == "postgresql":
            distance = func.bit_count(
                cast(Request.phash.op("#")(bindparam("probe", phash, type_=BigInteger)), BIT(64))
            )
            query = (
                base.add_columns(distance.label("distance"))
                .where(distance <= max_distance)
                .order_by("distance", Request.replied_at.desc())
                .limit(1)
            )
            row = (await session.execute(query)).first()
            return None if row is None else (row[0], row[1], int(row[3]))
        rows = (
            await session.execute(base.order_by(Request.replied_at.desc()).limit(candidates))
        ).all()
        best: tuple[uuid.UUID, dt.datetime | None, int] | None = None
        for request_id, replied_at, other in rows:
            if other is None:
                continue
            distance_now = hamming(phash, other)
            if distance_now <= max_distance and (best is None or distance_now < best[2]):
                best = (request_id, replied_at, distance_now)
        return best

    async def _cached(
        self,
        session: AsyncSession,
        request_id: uuid.UUID,
        replied_at: dt.datetime | None,
        match: Literal["exact", "phash"],
        distance: int,
    ) -> CachedVerdict | None:
        verdict = await session.get(VerdictRow, request_id)
        if verdict is None:
            return None
        return CachedVerdict(
            request_id=request_id,
            match=match,
            distance=distance,
            checked_at=as_utc(replied_at or dt.datetime.now(dt.UTC)),
            label=verdict.label,
            band=verdict.band,
            score=verdict.score,
            deciding_checker=verdict.deciding_checker,
            reply_text=verdict.reply_text,
            reply_language=verdict.reply_language,
        )

    async def save_cached(
        self,
        request_id: uuid.UUID,
        cached: CachedVerdict,
        fp: Fingerprint,
        duration_ms: int | None = None,
    ) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(VerdictRow).where(VerdictRow.request_id == request_id))
            session.add(
                VerdictRow(
                    request_id=request_id,
                    label=cached.label,
                    band=cached.band,
                    score=cached.score,
                    deciding_checker=cached.deciding_checker,
                    reply_text=cached.reply_text,
                    reply_language=cached.reply_language,
                )
            )
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    cache_of=cached.request_id,
                    sha256=fp.sha256,
                    phash=fp.phash,
                    caption_hash=fp.caption_hash,
                    cost_usd=0,
                    duration_ms=duration_ms,
                )
            )

    async def load_results(self, request_id: uuid.UUID) -> list[CheckerResult]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(CheckerRun).where(CheckerRun.request_id == request_id)
            )
            return [
                CheckerResult.model_validate(
                    {
                        "checker": row.checker,
                        "status": row.status,
                        "signal": row.signal,
                        "confidence": row.confidence,
                        "evidence": row.evidence,
                        "details": row.details,
                        "duration_ms": row.duration_ms,
                        "error": row.error,
                    }
                )
                for row in rows
            ]

    async def get_language(self, user_hash: str) -> str | None:
        async with self._sessions() as session:
            return await session.scalar(select(User.language).where(User.user_hash == user_hash))

    async def set_language(self, user_hash: str, language: str) -> None:
        # Two first-time writers can both see no row; the loser's insert fails and it retries
        # as an update.
        for attempt in (1, 2):
            try:
                async with self._sessions() as session, session.begin():
                    user = await session.get(User, user_hash)
                    if user is None:
                        session.add(User(user_hash=user_hash, language=language))
                    else:
                        user.language = language
                return
            except IntegrityError:
                if attempt == 2:
                    raise

    async def history(self, user_hash: str, limit: int) -> list[HistoryItem]:
        query = (
            select(Request.created_at, Request.status, VerdictRow.label, VerdictRow.band)
            .join(VerdictRow, VerdictRow.request_id == Request.id, isouter=True)
            .where(Request.user_hash == user_hash)
            .order_by(Request.created_at.desc())
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        return [HistoryItem(as_utc(r[0]), r[1], r[2], r[3]) for r in rows]

    async def load_details(self, request_id: uuid.UUID, user_hash: str) -> DetailsData | None:
        async with self._sessions() as session:
            request = await session.get(Request, request_id)
            verdict = await session.get(VerdictRow, request_id)
            if request is None or request.user_hash != user_hash or verdict is None:
                return None
            source = request.cache_of or request_id
        return DetailsData(
            verdict.label,
            verdict.band,
            verdict.score,
            verdict.reply_language,
            await self.load_results(source),
        )

    async def save_feedback(self, request_id: uuid.UUID, user_hash: str, helpful: bool) -> bool:
        for _attempt in range(2):
            try:
                async with self._sessions() as session, session.begin():
                    request = await session.get(Request, request_id)
                    if request is None or request.user_hash != user_hash:
                        return False
                    row = await session.get(Feedback, (request_id, user_hash))
                    if row is None:
                        session.add(
                            Feedback(request_id=request_id, user_hash=user_hash, helpful=helpful)
                        )
                    else:
                        row.helpful = helpful
                        row.updated_at = dt.datetime.now(dt.UTC)
                return True
            except IntegrityError:  # a concurrent vote inserted the row first: update it
                continue
        return False

    async def delete_user(self, user_hash: str) -> int:
        async with self._sessions() as session, session.begin():
            return await delete_user_rows(session, user_hash)

    async def purge_older_than(self, cutoff: dt.datetime) -> int:
        async with self._sessions() as session, session.begin():
            return await purge_rows(session, cutoff)
