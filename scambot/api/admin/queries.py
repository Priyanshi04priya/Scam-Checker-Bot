"""Read-only statistics for the admin pages and the error alert.

Every number is computed in SQL; no message text, chat ID or checker detail leaves this module.
Users appear only as hashes.
"""

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy import Integer, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.locales import IST
from scambot.core.models import CheckerRun, Feedback, Request, User, VerdictRow

TOP_ERRORS = 3


def day_bucket(dialect: str, column: Any) -> Any:
    """The India-time calendar day of a UTC timestamp column."""
    if dialect == "postgresql":
        return func.date(func.timezone("Asia/Kolkata", column))
    return func.date(column, "+5 hours", "+30 minutes")


@dataclass(frozen=True)
class CheckerErrorRate:
    checker: str
    total: int
    errors: int
    top_errors: list[str]

    @property
    def rate(self) -> float:
        return self.errors / self.total if self.total else 0.0


@dataclass(frozen=True)
class VerdictShare:
    label: str
    count: int
    share: float


@dataclass(frozen=True)
class FeedbackSummary:
    helpful: int = 0
    wrong: int = 0
    wrong_by_label: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # label -> (wrong, all)

    @property
    def total(self) -> int:
        return self.helpful + self.wrong


@dataclass(frozen=True)
class Latency:
    overall_ms: float | None = None
    fresh_ms: float | None = None
    cache_ms: float | None = None
    by_kind: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Overview:
    days: int
    total: int
    per_day: list[tuple[str, int]]
    verdicts: list[VerdictShare]
    latency: Latency
    checkers: list[CheckerErrorRate]
    checkers_24h: dict[str, CheckerErrorRate]
    feedback: FeedbackSummary
    average_cost: Decimal
    blocked_users: int


@dataclass(frozen=True)
class UserRow:
    user_hash: str
    requests_24h: int
    blocked: bool


def ist_day_start(now: dt.datetime, days_back: int) -> dt.datetime:
    """Midnight India time ``days_back`` days before today's, as a UTC instant."""
    local = now.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    return (local - dt.timedelta(days=days_back)).astimezone(dt.UTC)


def _average(value: Any) -> float | None:
    return None if value is None else float(value)


class StatsStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def checker_error_rates(self, since: dt.datetime) -> list[CheckerErrorRate]:
        """Runs and failures per checker among requests created since ``since``."""
        errored = func.sum(case((CheckerRun.status == "error", 1), else_=0))
        totals = (
            select(CheckerRun.checker, func.count(), errored)
            .join(Request, Request.id == CheckerRun.request_id)
            .where(Request.created_at >= since)
            .group_by(CheckerRun.checker)
            .order_by(CheckerRun.checker)
        )
        codes = (
            select(CheckerRun.checker, CheckerRun.error, func.count().label("n"))
            .join(Request, Request.id == CheckerRun.request_id)
            .where(
                Request.created_at >= since,
                CheckerRun.status == "error",
                CheckerRun.error.is_not(None),
            )
            .group_by(CheckerRun.checker, CheckerRun.error)
            .order_by(func.count().desc(), CheckerRun.error)
        )
        async with self._sessions() as session:
            rows = (await session.execute(totals)).all()
            code_rows = (await session.execute(codes)).all()
        top: dict[str, list[str]] = defaultdict(list)
        for checker, code, _n in code_rows:
            if len(top[checker]) < TOP_ERRORS:
                top[checker].append(str(code)[:60])
        return [
            CheckerErrorRate(checker, int(total), int(errors or 0), top[checker])
            for checker, total, errors in rows
        ]

    async def overview(self, days: int, now: dt.datetime) -> Overview:
        since = ist_day_start(now, days - 1)
        async with self._sessions() as session:
            dialect = session.get_bind().dialect.name
            per_day = await self._per_day(session, dialect, since, now, days)
            verdicts = await self._verdicts(session, since)
            latency = await self._latency(session, since)
            feedback = await self._feedback(session, since)
            cost = await session.scalar(
                select(func.avg(Request.cost_usd)).where(
                    Request.created_at >= since, Request.status == "replied"
                )
            )
            blocked = await session.scalar(
                select(func.count()).select_from(User).where(User.blocked.is_(True))
            )
        recent = await self.checker_error_rates(now - dt.timedelta(hours=24))
        return Overview(
            days=days,
            total=sum(count for _day, count in per_day),
            per_day=per_day,
            verdicts=verdicts,
            latency=latency,
            checkers=await self.checker_error_rates(since),
            checkers_24h={item.checker: item for item in recent},
            feedback=feedback,
            average_cost=Decimal(str(cost)) if cost is not None else Decimal(0),
            blocked_users=int(blocked or 0),
        )

    @staticmethod
    async def _per_day(
        session: AsyncSession, dialect: str, since: dt.datetime, now: dt.datetime, days: int
    ) -> list[tuple[str, int]]:
        bucket = day_bucket(dialect, Request.created_at)
        rows = (
            await session.execute(
                select(bucket, func.count()).where(Request.created_at >= since).group_by(bucket)
            )
        ).all()
        counts = {str(day): int(count) for day, count in rows}
        today = now.astimezone(IST).date()
        series = [(today - dt.timedelta(days=back)).isoformat() for back in range(days - 1, -1, -1)]
        return [(day, counts.get(day, 0)) for day in series]

    @staticmethod
    async def _verdicts(session: AsyncSession, since: dt.datetime) -> list[VerdictShare]:
        rows = (
            await session.execute(
                select(VerdictRow.label, func.count())
                .join(Request, Request.id == VerdictRow.request_id)
                .where(Request.created_at >= since)
                .group_by(VerdictRow.label)
                .order_by(func.count().desc(), VerdictRow.label)
            )
        ).all()
        total = sum(int(count) for _label, count in rows)
        return [
            VerdictShare(label, int(count), int(count) / total if total else 0.0)
            for label, count in rows
        ]

    @staticmethod
    async def _latency(session: AsyncSession, since: dt.datetime) -> Latency:
        base = (Request.created_at >= since, Request.status == "replied")
        overall = await session.scalar(select(func.avg(Request.duration_ms)).where(*base))
        fresh = await session.scalar(
            select(func.avg(Request.duration_ms)).where(*base, Request.cache_of.is_(None))
        )
        cached = await session.scalar(
            select(func.avg(Request.duration_ms)).where(*base, Request.cache_of.is_not(None))
        )
        by_kind = (
            await session.execute(
                select(Request.kind, func.avg(Request.duration_ms))
                .where(*base, Request.duration_ms.is_not(None))
                .group_by(Request.kind)
                .order_by(Request.kind)
            )
        ).all()
        return Latency(
            _average(overall),
            _average(fresh),
            _average(cached),
            {kind: float(avg) for kind, avg in by_kind},
        )

    @staticmethod
    async def _feedback(session: AsyncSession, since: dt.datetime) -> FeedbackSummary:
        wrong = func.sum(case((Feedback.helpful.is_(False), 1), else_=0))
        right = func.sum(case((Feedback.helpful.is_(True), 1), else_=0))
        rows = (
            await session.execute(
                select(VerdictRow.label, cast(right, Integer), cast(wrong, Integer))
                .join(VerdictRow, VerdictRow.request_id == Feedback.request_id)
                .where(Feedback.created_at >= since)
                .group_by(VerdictRow.label)
                .order_by(VerdictRow.label)
            )
        ).all()
        return FeedbackSummary(
            helpful=sum(int(r[1] or 0) for r in rows),
            wrong=sum(int(r[2] or 0) for r in rows),
            wrong_by_label={r[0]: (int(r[2] or 0), int(r[1] or 0) + int(r[2] or 0)) for r in rows},
        )

    async def recent_users(self, limit: int, now: dt.datetime) -> list[UserRow]:
        """Users with requests in the last 24 hours, busiest first."""
        count = func.count(Request.id)
        query = (
            select(Request.user_hash, count, func.coalesce(User.blocked, False))
            .join(User, User.user_hash == Request.user_hash, isouter=True)
            .where(Request.created_at >= now - dt.timedelta(hours=24))
            .group_by(Request.user_hash, User.blocked)
            .order_by(count.desc(), Request.user_hash)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        return [UserRow(str(h), int(n), bool(blocked)) for h, n, blocked in rows]

    async def blocked_users(self, limit: int = 100) -> list[str]:
        query = select(User.user_hash).where(User.blocked.is_(True)).order_by(User.user_hash)
        async with self._sessions() as session:
            return [str(h) for h in await session.scalars(query.limit(limit))]

    async def find_users(self, prefix: str) -> list[str]:
        """Up to two user hashes starting with ``prefix``; two show that it is ambiguous."""
        query = select(User.user_hash).where(User.user_hash.startswith(prefix)).limit(2)
        async with self._sessions() as session:
            return [str(h) for h in await session.scalars(query)]
