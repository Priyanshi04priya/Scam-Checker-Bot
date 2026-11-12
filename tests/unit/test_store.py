"""Tests for request persistence: SQLite (always), PostgreSQL (opt-in) and the in-memory fake."""

import dataclasses
import datetime as dt
import os
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import event, func, select, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry

from scambot.core.contracts import CheckerResult, Evidence, ExecutedCheck, Verdict
from scambot.core.db import create_engine, create_session_factory
from scambot.core.models import Base, CheckerRun, Request, User, VerdictRow
from scambot.core.store import NewRequest, RequestStore, SqlRequestStore, StoredResult
from tests.fakes import InMemoryStore

POSTGRES_URL = os.environ.get("SCAMBOT_TEST_DATABASE_URL", "")

NEW = NewRequest(
    job_id="telegram:1",
    user_hash="u" * 64,
    channel="telegram",
    kind="text",
    sha256="a" * 64,
    language="hi",
)


def result(confidence: float = 0.8) -> StoredResult:
    check = CheckerResult(
        checker="analyze_scam_patterns",
        status="ok",
        signal="scam",
        confidence=confidence,
        duration_ms=12,
        evidence=[Evidence(summary='threat: "बिजली काट दी जाएगी"', source="rules")],
        details={"signals": ["electricity_disconnection"], "template": "बिजली"},
    )
    return StoredResult(
        checks=[ExecutedCheck(result=check, input_hash="h" * 64)],
        verdict=Verdict(
            label="likely_scam", band="high", score=94, deciding_checker="analyze_scam_patterns"
        ),
        reply_text="🔴 उच्च जोखिम · संभावित धोखाधड़ी",
        reply_language="hi",
        claims=["बिजली का बिल बकाया है"],
        language="hi",
        cost_usd=Decimal("0.001234"),
    )


async def _sql_store(url: str, *, sqlite: bool) -> AsyncIterator[SqlRequestStore]:
    engine = create_engine(SecretStr(url))
    if sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _foreign_keys(connection: DBAPIConnection, _record: ConnectionPoolEntry) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield SqlRequestStore(create_session_factory(engine))
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.fixture
async def sqlite_store(tmp_path: Path) -> AsyncIterator[SqlRequestStore]:
    async for store in _sql_store(f"sqlite+aiosqlite:///{tmp_path}/t.db", sqlite=True):
        yield store


@pytest.fixture(params=["sqlite", "memory"])
async def any_store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[RequestStore]:
    if request.param == "memory":
        yield InMemoryStore()
        return
    async for store in _sql_store(f"sqlite+aiosqlite:///{tmp_path}/t.db", sqlite=True):
        yield store


async def test_start_is_idempotent(any_store: RequestStore) -> None:
    first = await any_store.start(NEW)
    second = await any_store.start(NEW)
    other = await any_store.start(
        NewRequest("telegram:2", NEW.user_hash, "telegram", "text", None, None)
    )

    assert first.created
    assert not second.created
    assert first.id == second.id
    assert first.status == second.status == "received"
    assert other.id != first.id


async def test_save_and_status_round_trip(any_store: RequestStore) -> None:
    started = await any_store.start(NEW)

    await any_store.save(started.id, result(0.7))
    await any_store.save(started.id, result(0.9))  # a retry replaces the earlier rows
    replied_at = dt.datetime(2026, 11, 12, 10, 0, tzinfo=dt.UTC)
    await any_store.set_status(started.id, "replied", at=replied_at)

    again = await any_store.start(NEW)
    assert again.status == "replied"


async def _counts(store: SqlRequestStore) -> tuple[int, int, int]:
    async with store._sessions() as session:
        runs = await session.scalar(select(func.count()).select_from(CheckerRun))
        verdicts = await session.scalar(select(func.count()).select_from(VerdictRow))
        users = await session.scalar(select(func.count()).select_from(User))
    return runs or 0, verdicts or 0, users or 0


async def test_sql_rows(sqlite_store: SqlRequestStore) -> None:
    started = await sqlite_store.start(NEW)
    await sqlite_store.save(started.id, result(0.7))
    await sqlite_store.save(started.id, result(0.9))
    await sqlite_store.set_status(started.id, "replied")

    assert await _counts(sqlite_store) == (1, 1, 1)
    async with sqlite_store._sessions() as session:
        row = await session.get(Request, started.id)
        run = (await session.scalars(select(CheckerRun))).one()
        verdict = (await session.scalars(select(VerdictRow))).one()
    assert row is not None
    assert row.status == "replied"
    assert row.replied_at is not None
    assert row.claims == ["बिजली का बिल बकाया है"]
    assert row.cost_usd == Decimal("0.001234")
    assert row.language == "hi"
    assert run.confidence == 0.9
    assert run.evidence == [
        {"summary": 'threat: "बिजली काट दी जाएगी"', "url": None, "source": "rules", "date": None}
    ]
    assert run.details["template"] == "बिजली"
    assert verdict.reply_text.startswith("🔴")


async def test_unknown_status_is_rejected(sqlite_store: SqlRequestStore) -> None:
    started = await sqlite_store.start(NEW)

    with pytest.raises(Exception, match="CHECK constraint"):
        await sqlite_store.set_status(started.id, "lost")  # type: ignore[arg-type]


async def test_deleting_a_user_cascades(sqlite_store: SqlRequestStore) -> None:
    started = await sqlite_store.start(NEW)
    await sqlite_store.save(started.id, result())

    async with sqlite_store._sessions() as session, session.begin():
        await session.execute(text("DELETE FROM users"))

    assert await _counts(sqlite_store) == (0, 0, 0)


async def test_concurrent_insert_is_retried(sqlite_store: SqlRequestStore) -> None:
    original = sqlite_store._insert
    calls = 0

    async def racing(request: NewRequest) -> Request:
        nonlocal calls
        calls += 1
        if calls == 1:
            await original(request)  # another worker wins the race...
        return await original(request)  # ...so this insert hits the unique constraint

    sqlite_store._insert = racing  # type: ignore[method-assign]

    started = await sqlite_store.start(NEW)

    assert not started.created
    assert calls == 1


async def test_in_memory_store_can_fail() -> None:
    store = InMemoryStore(fail=True)

    with pytest.raises(ConnectionError):
        await store.start(NEW)
    with pytest.raises(ConnectionError):
        await store.save(uuid.uuid4(), result())
    with pytest.raises(ConnectionError):
        await store.set_status(uuid.uuid4(), "failed")


@pytest.mark.skipif(not POSTGRES_URL, reason="SCAMBOT_TEST_DATABASE_URL is not set")
class TestPostgres:
    @pytest.fixture
    async def pg_store(self) -> AsyncIterator[SqlRequestStore]:
        async for store in _sql_store(POSTGRES_URL, sqlite=False):
            yield store

    async def test_round_trip(self, pg_store: SqlRequestStore) -> None:
        started = await pg_store.start(NEW)
        await pg_store.save(started.id, result(0.7))
        await pg_store.save(started.id, result(0.9))
        await pg_store.set_status(started.id, "replied")

        assert (await pg_store.start(NEW)).status == "replied"
        assert await _counts(pg_store) == (1, 1, 1)


async def test_save_without_language_keeps_the_start_language(
    sqlite_store: SqlRequestStore,
) -> None:
    started = await sqlite_store.start(NEW)
    await sqlite_store.save(started.id, dataclasses.replace(result(), language=None))

    async with sqlite_store._sessions() as session:
        row = await session.get(Request, started.id)
    assert row is not None
    assert row.language == "hi"
