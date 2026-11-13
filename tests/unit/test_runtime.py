"""Runtime settings: precedence, caching and failing open."""

import datetime as dt

import pytest
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.config import AppConfig
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults, SqlSettingsStore
from scambot.core.settings import Settings
from tests.fakes import InMemorySettingsStore

DEFAULTS = RuntimeDefaults(rate_per_hour=20, window_days=7)


def runtime(store: InMemorySettingsStore, redis: ArqRedis) -> RuntimeConfig:
    return RuntimeConfig(store, redis, DEFAULTS, 60)


async def test_defaults_apply_without_overrides(redis: ArqRedis) -> None:
    config = runtime(InMemorySettingsStore(), redis)

    assert await config.rate_limit() == 20
    assert await config.verdict_window() == dt.timedelta(days=7)
    assert await config.is_blocked("u") is False


async def test_the_settings_table_overrides_the_defaults(redis: ArqRedis) -> None:
    store = InMemorySettingsStore({"rate_limit_per_hour": 5, "cache_window_days": 0.5})
    config = runtime(store, redis)

    assert await config.rate_limit() == 5
    assert await config.verdict_window() == dt.timedelta(hours=12)


async def test_values_are_cached_for_a_minute_and_can_be_invalidated(redis: ArqRedis) -> None:
    store = InMemorySettingsStore({"rate_limit_per_hour": 5})
    config = runtime(store, redis)
    assert await config.rate_limit() == 5

    await store.set("rate_limit_per_hour", 9)
    assert await config.rate_limit() == 5
    assert 0 < await redis.ttl("cfg:rate_limit_per_hour") <= 60

    await config.invalidate("rate_limit_per_hour")
    assert await config.rate_limit() == 9


async def test_a_missing_override_is_cached_too(redis: ArqRedis) -> None:
    store = InMemorySettingsStore()
    config = runtime(store, redis)

    await config.rate_limit()
    await config.rate_limit()

    assert store.reads == 1


@pytest.mark.parametrize("bad", [0, -3, "ten", None, True, []])
async def test_invalid_values_are_ignored(redis: ArqRedis, bad: object) -> None:
    config = runtime(InMemorySettingsStore({"rate_limit_per_hour": bad}), redis)

    assert await config.rate_limit() == 20


async def test_a_database_error_falls_back_to_the_defaults(redis: ArqRedis) -> None:
    config = runtime(InMemorySettingsStore(fail=True), redis)

    assert await config.rate_limit() == 20
    assert await config.verdict_window() == dt.timedelta(days=7)
    assert await config.is_blocked("u") is False


async def test_blocked_users_are_cached_and_unblocking_needs_an_invalidate(redis: ArqRedis) -> None:
    store = InMemorySettingsStore(blocked={"u"})
    config = runtime(store, redis)
    assert await config.is_blocked("u") is True

    await store.set_blocked("u", False)
    assert await config.is_blocked("u") is True

    await config.invalidate("blocked:u")
    assert await config.is_blocked("u") is False


async def test_a_redis_error_falls_back_to_the_defaults() -> None:
    class Broken:
        async def get(self, *_args: object) -> None:
            raise ConnectionError

    config = RuntimeConfig(InMemorySettingsStore(), Broken(), DEFAULTS, 60)  # type: ignore[arg-type]

    assert await config.rate_limit() == 20
    assert await config.is_blocked("u") is False


def test_the_environment_beats_the_yaml(app_config: AppConfig) -> None:
    from_yaml = RuntimeDefaults.build(app_config.thresholds, Settings(_env_file=None))
    from_env = RuntimeDefaults.build(
        app_config.thresholds,
        Settings(_env_file=None, rate_limit_per_hour=3, cache_window_days=0.25),
    )

    assert (from_yaml.rate_per_hour, from_yaml.window_days) == (20, 7)
    assert (from_env.rate_per_hour, from_env.window_days) == (3, 0.25)


async def test_describe_reports_value_and_origin(redis: ArqRedis) -> None:
    store = InMemorySettingsStore({"rate_limit_per_hour": 5})
    config = RuntimeConfig(store, redis, RuntimeDefaults(20, 7, "env", "file"), 60)

    described = await config.describe()

    assert described["rate_limit_per_hour"] == {"value": 5, "origin": "table"}
    assert described["cache_window_days"] == {"value": 7, "origin": "file"}
    await store.delete("rate_limit_per_hour")
    assert (await config.describe())["rate_limit_per_hour"] == {"value": 20, "origin": "env"}


async def test_the_sql_store_can_delete_a_setting(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    store = SqlSettingsStore(sessions)
    await store.set("rate_limit_per_hour", 3)
    await store.delete("rate_limit_per_hour")
    await store.delete("rate_limit_per_hour")

    assert await store.get("rate_limit_per_hour") is None
