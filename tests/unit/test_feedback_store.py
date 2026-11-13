"""User data kept for buttons and commands: language, history, details and feedback."""

import datetime as dt
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from scambot.core.contracts import CheckerResult, ExecutedCheck, Verdict
from scambot.core.db import create_engine, create_session_factory
from scambot.core.models import Base, Feedback, User
from scambot.core.store import CachedVerdict, Fingerprint, NewRequest, SqlRequestStore, StoredResult
from tests.fakes import InMemoryStore

USER = "u" * 64
OTHER = "o" * 64


def stored(duration: int | None = 1200) -> StoredResult:
    check = CheckerResult(
        checker="analyze_scam_patterns", status="ok", signal="scam", confidence=0.9, duration_ms=3
    )
    return StoredResult(
        checks=[ExecutedCheck(result=check, input_hash="h" * 64)],
        verdict=Verdict(label="likely_scam", band="high", score=90),
        reply_text="text",
        reply_language="hi",
        claims=[],
        language="hi",
        cost_usd=0,  # type: ignore[arg-type]
        duration_ms=duration,
    )


@pytest.fixture(params=["sql", "memory"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[object]:
    if request.param == "memory":
        yield InMemoryStore()
        return
    engine = create_engine(SecretStr(f"sqlite+aiosqlite:///{tmp_path / 'f.db'}"))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield SqlRequestStore(create_session_factory(engine))
    await engine.dispose()


async def make(store: object, job: str, user: str = USER, *, done: bool = True) -> uuid.UUID:
    started = await store.start(  # type: ignore[attr-defined]
        NewRequest(job, user, "telegram", "text", "a" * 64, "en")
    )
    if done:
        await store.save(started.id, stored())  # type: ignore[attr-defined]
        await store.set_status(started.id, "replied")  # type: ignore[attr-defined]
    return started.id  # type: ignore[no-any-return]


async def test_language_is_kept_per_user(store: object) -> None:
    assert await store.get_language(USER) is None  # type: ignore[attr-defined]
    await store.set_language(USER, "hi")  # type: ignore[attr-defined]
    await store.set_language(USER, "hi-Latn")  # type: ignore[attr-defined]
    assert await store.get_language(USER) == "hi-Latn"  # type: ignore[attr-defined]


async def test_a_vote_can_be_changed_and_only_the_owner_can_cast_it(store: object) -> None:
    request_id = await make(store, "t:1")

    assert await store.save_feedback(request_id, USER, True)  # type: ignore[attr-defined]
    assert await store.save_feedback(request_id, USER, False)  # type: ignore[attr-defined]
    assert not await store.save_feedback(request_id, OTHER, True)  # type: ignore[attr-defined]
    assert not await store.save_feedback(uuid.uuid4(), USER, True)  # type: ignore[attr-defined]
    if isinstance(store, SqlRequestStore):
        async with store._sessions() as session:
            rows = (await session.scalars(select(Feedback))).all()
        assert [(r.user_hash, r.helpful) for r in rows] == [(USER, False)]
    else:
        assert store.feedback == {(request_id, USER): False}  # type: ignore[attr-defined]


async def test_history_is_newest_first_limited_and_has_no_text(store: object) -> None:
    for i in range(12):
        await make(store, f"t:{i}", done=i != 3)
        await make(store, f"x:{i}", OTHER)

    items = await store.history(USER, 10)  # type: ignore[attr-defined]

    assert len(items) == 10
    assert items == sorted(items, key=lambda item: item.created_at, reverse=True)
    assert {item.status for item in items} <= {"replied", "received"}
    assert all(item.label in {None, "likely_scam"} for item in items)
    assert not hasattr(items[0], "text")


async def test_details_belong_to_the_owner_and_follow_a_reused_answer(store: object) -> None:
    original = await make(store, "t:1")
    second = await make(store, "t:2", done=False)
    cached = CachedVerdict(
        original, "exact", 0, dt.datetime.now(dt.UTC), "likely_scam", "high", 90, None, "t", "hi"
    )
    await store.save_cached(second, cached, Fingerprint("text", "a" * 64, None, None), 50)  # type: ignore[attr-defined]

    own = await store.load_details(second, USER)  # type: ignore[attr-defined]

    assert own is not None
    assert [r.checker for r in own.results] == ["analyze_scam_patterns"]
    assert await store.load_details(second, OTHER) is None  # type: ignore[attr-defined]
    assert await store.load_details(uuid.uuid4(), USER) is None  # type: ignore[attr-defined]


async def test_deleting_removes_feedback_and_the_duration_is_stored(store: object) -> None:
    keep = await make(store, "t:1")
    await store.save_feedback(keep, USER, True)  # type: ignore[attr-defined]
    if isinstance(store, SqlRequestStore):
        async with store._sessions() as session:
            row = await session.scalar(
                select(
                    func.max(
                        __import__("scambot.core.models", fromlist=["Request"]).Request.duration_ms
                    )
                )
            )
        assert row == 1200
    else:
        assert store.requests["t:1"].duration_ms == 1200  # type: ignore[attr-defined]

    assert await store.delete_user(USER) == 1  # type: ignore[attr-defined]

    if isinstance(store, SqlRequestStore):
        async with store._sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Feedback)) == 0
    else:
        assert store.feedback == {}  # type: ignore[attr-defined]


async def test_purging_removes_feedback(store: object) -> None:
    request_id = await make(store, "t:1")
    await store.save_feedback(request_id, USER, True)  # type: ignore[attr-defined]

    future = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
    assert await store.purge_older_than(future) == 1  # type: ignore[attr-defined]

    if isinstance(store, SqlRequestStore):
        async with store._sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Feedback)) == 0
    else:
        assert store.feedback == {}  # type: ignore[attr-defined]


async def test_two_first_time_language_writers_do_not_clash(
    store: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not isinstance(store, SqlRequestStore):
        pytest.skip("only the database store can race")
    original = AsyncSession.get
    calls = 0

    async def racing(self: AsyncSession, entity: Any, ident: Any, **kw: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:  # another writer inserts the row between the read and the insert
            async with store._sessions() as other, other.begin():
                other.add(User(user_hash=USER, language="en"))
            return None
        return await original(self, entity, ident, **kw)

    monkeypatch.setattr(AsyncSession, "get", racing)

    await store.set_language(USER, "hi")

    assert await store.get_language(USER) == "hi"
