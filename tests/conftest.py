"""Shared pytest fixtures."""

from collections.abc import AsyncIterator, Iterator

import pytest
import structlog
from arq.connections import ArqRedis
from fakeredis import FakeAsyncRedis, FakeServer

from scambot.core.locales import Locales, load_locales
from scambot.core.settings import Settings, get_settings
from tests.fakes import FakeAdapter


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every settings variable from the environment, stop the project ``.env`` from
    being read and clear the settings cache."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Restore structlog's default configuration after each test."""
    yield
    structlog.reset_defaults()


@pytest.fixture
async def redis() -> AsyncIterator[ArqRedis]:
    """A queue-capable Redis client backed by a private in-memory server."""
    client = ArqRedis(connection_pool=FakeAsyncRedis(server=FakeServer()).connection_pool)
    yield client
    # The pool was passed in explicitly, so the client would otherwise leave it open.
    await client.aclose(close_connection_pool=True)


@pytest.fixture(scope="session")
def locales() -> Locales:
    return load_locales()


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()
