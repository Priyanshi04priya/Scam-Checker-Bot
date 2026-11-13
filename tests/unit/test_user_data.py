"""Deleting one user's records and purging old ones."""

import datetime as dt
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
from arq.connections import ArqRedis
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.contracts import CheckerResult, ExecutedCheck, Verdict
from scambot.core.db import create_engine, create_session_factory
from scambot.core.models import Base, CheckerRun, Request, User, VerdictRow
from scambot.core.redis import clear_user_keys, hit_rate, rate_key
from scambot.core.runtime import SqlSettingsStore
from scambot.core.store import (
    CachedVerdict,
    Fingerprint,
    NewRequest,
    RequestStore,
    SqlRequestStore,
    StoredResult,
)
from tests.fakes import InMemoryStore

NOW = dt.datetime(2026, 11, 13, 12, 0, tzinfo=dt.UTC)


def stored() -> StoredResult:
    check = CheckerResult(
        checker="analyze_scam_patterns", status="ok", signal="scam", confidence=0.9, duration_ms=1
    )
    return StoredResult(
        checks=[ExecutedCheck(result=check, input_hash="h" * 64)],
        verdict=Verdict(label="likely_scam", band="high", score=90),
        reply_text="r",
        reply_language="en",
        claims=[],
        language="en",
        cost_usd=Decimal(0),
    )


def sessions_of(store: RequestStore) -> async_sessionmaker[AsyncSession]:
    assert isinstance(store, SqlRequestStore)
    return store._sessions


@pytest.fixture(params=["sqlite", "memory"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[RequestStore]:
    if request.param == "memory":
        yield InMemoryStore()
        return
    engine = create_engine(SecretStr(f"sqlite+aiosqlite:///{tmp_path}/t.db"))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield SqlRequestStore(create_session_factory(engine))
    await engine.dispose()


async def add(store: RequestStore, job: str, user: str, age_days: float = 1) -> None:
    started = await store.start(NewRequest(job, user, "telegram", "text", job * 8, "en"))
    await store.save(started.id, stored())
    await store.set_status(started.id, "replied", at=NOW)
    created = NOW - dt.timedelta(days=age_days)
    if isinstance(store, InMemoryStore):
        store.requests[job].created_at = created
        return
    async with sessions_of(store)() as session, session.begin():
        await session.execute(
            update(Request).where(Request.job_id == job).values(created_at=created)
        )
        await session.execute(
            update(User)
            .where(User.user_hash == user, User.created_at > created)
            .values(created_at=created)
        )


async def counts(store: RequestStore) -> tuple[int, int, int, int]:
    """Requests, checker runs, verdicts and users left."""
    if isinstance(store, InMemoryStore):
        rows = list(store.requests.values())
        return (
            len(rows),
            sum(len(r.checks) for r in rows),
            sum(r.verdict is not None for r in rows),
            len(store.users),
        )
    assert isinstance(store, SqlRequestStore)
    async with sessions_of(store)() as session:
        found = [
            await session.scalar(select(func.count()).select_from(model)) or 0
            for model in (Request, CheckerRun, VerdictRow, User)
        ]
        return found[0], found[1], found[2], found[3]


async def test_delete_user_removes_everything_of_that_user_only(store: RequestStore) -> None:
    await add(store, "a1", "alice")
    await add(store, "a2", "alice")
    await add(store, "b1", "bob")

    removed = await store.delete_user("alice")

    assert removed == 2
    assert await counts(store) == (1, 1, 1, 1)


async def test_delete_user_for_an_unknown_user_removes_nothing(store: RequestStore) -> None:
    await add(store, "a1", "alice")

    assert await store.delete_user("nobody") == 0
    assert await counts(store) == (1, 1, 1, 1)


async def test_delete_user_frees_requests_that_reused_a_deleted_one(store: RequestStore) -> None:
    await add(store, "a1", "alice")
    await add(store, "b1", "bob")
    if isinstance(store, InMemoryStore):
        first, second = store.requests["a1"], store.requests["b1"]
        first_id, second_id = first.id, second.id
    else:
        async with sessions_of(store)() as session:
            first_id, second_id = [
                (
                    await session.execute(select(Request.id).where(Request.job_id == job))
                ).scalar_one()
                for job in ("a1", "b1")
            ]
    cached = CachedVerdict(first_id, "exact", 0, NOW, "likely_scam", "high", 90, None, "r", "en")
    await store.save_cached(second_id, cached, Fingerprint("text", "a1" * 32, None, None))

    assert await store.delete_user("alice") == 1

    assert await counts(store) == (1, 1, 1, 1)


async def test_a_blocked_user_keeps_the_blocking_row(store: RequestStore, tmp_path: Path) -> None:
    await add(store, "a1", "alice")
    if isinstance(store, InMemoryStore):
        store.blocked.add("alice")
    else:
        await SqlSettingsStore(sessions_of(store)).set_blocked("alice", True)

    assert await store.delete_user("alice") == 1

    assert await counts(store) == (0, 0, 0, 1)


async def test_purge_removes_old_requests_with_their_children(store: RequestStore) -> None:
    await add(store, "old", "alice", age_days=91)
    await add(store, "new", "alice", age_days=89)
    await add(store, "gone", "carol", age_days=100)

    removed = await store.purge_older_than(NOW - dt.timedelta(days=90))

    assert removed == 2
    assert await counts(store) == (1, 1, 1, 1)


async def test_purge_batches_through_many_rows(
    store: RequestStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scambot.core.purge.PURGE_BATCH", 2)
    monkeypatch.setattr("scambot.core.purge.CHUNK", 2)
    for i in range(5):
        await add(store, f"old{i}", "alice", age_days=120)

    assert await store.purge_older_than(NOW - dt.timedelta(days=90)) == 5
    assert await counts(store) == (0, 0, 0, 0)


async def test_clear_user_keys_removes_the_block_flag_but_keeps_counters(redis: ArqRedis) -> None:
    await hit_rate(redis, "alice", NOW, 3600)
    await hit_rate(redis, "alice", NOW + dt.timedelta(hours=1), 3600)
    await hit_rate(redis, "bob", NOW, 3600)
    await redis.set("cfg:blocked:alice", "false")

    assert await clear_user_keys(redis, "alice") == 1

    assert await redis.exists(rate_key("bob", NOW)) == 1
    assert await redis.exists(rate_key("alice", NOW)) == 1
    assert await redis.exists("cfg:blocked:alice") == 0
