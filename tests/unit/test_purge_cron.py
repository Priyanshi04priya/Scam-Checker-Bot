"""The nightly retention purge."""

import datetime as dt
from decimal import Decimal
from pathlib import Path

from pydantic import SecretStr
from sqlalchemy import func, select, update

from scambot.core.db import create_engine, create_session_factory
from scambot.core.models import Base, Request, User
from scambot.core.store import NewRequest, SqlRequestStore, StoredResult
from scambot.worker.cron import PURGE_KEY, PurgeDeps, purge_retention


async def test_requests_older_than_the_retention_period_are_purged(tmp_path: Path) -> None:
    engine = create_engine(SecretStr(f"sqlite+aiosqlite:///{tmp_path}/t.db"))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = create_session_factory(engine)
    store = SqlRequestStore(sessions)
    now = dt.datetime.now(dt.UTC)
    for job, user, age in (("old", "u1", 91), ("new", "u2", 89)):
        started = await store.start(NewRequest(job, user, "telegram", "text", job, "en"))
        await store.save(
            started.id,
            StoredResult([], None, "r", "en", [], "en", Decimal(0)),
        )
        created = now - dt.timedelta(days=age)
        async with sessions() as session, session.begin():
            await session.execute(
                update(Request).where(Request.id == started.id).values(created_at=created)
            )
            await session.execute(
                update(User).where(User.user_hash == user).values(created_at=created)
            )

    removed = await purge_retention({PURGE_KEY: PurgeDeps(store, 90)})

    assert removed == 1
    async with sessions() as session:
        assert (await session.scalars(select(Request.job_id))).all() == ["new"]
        assert await session.scalar(select(func.count()).select_from(User)) == 1
    await engine.dispose()
