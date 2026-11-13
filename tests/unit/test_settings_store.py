"""The settings table and the block list."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import SecretStr

from scambot.core.db import create_engine, create_session_factory
from scambot.core.models import Base
from scambot.core.runtime import SettingsStore, SqlSettingsStore
from tests.fakes import InMemorySettingsStore


@pytest.fixture(params=["sqlite", "memory"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[SettingsStore]:
    if request.param == "memory":
        yield InMemorySettingsStore()
        return
    engine = create_engine(SecretStr(f"sqlite+aiosqlite:///{tmp_path}/t.db"))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield SqlSettingsStore(create_session_factory(engine))
    await engine.dispose()


async def test_get_set_upsert(store: SettingsStore) -> None:
    assert await store.get("rate_limit_per_hour") is None

    await store.set("rate_limit_per_hour", 5)
    await store.set("rate_limit_per_hour", 7.5)

    assert await store.get("rate_limit_per_hour") == 7.5


async def test_block_and_unblock_create_the_user_when_missing(store: SettingsStore) -> None:
    assert await store.is_blocked("u") is False

    await store.set_blocked("u", True)
    assert await store.is_blocked("u") is True

    await store.set_blocked("u", False)
    assert await store.is_blocked("u") is False
