"""The hourly counter and the intake gate."""

import datetime as dt

import pytest
from arq.connections import ArqRedis

from scambot.api.gate import Admission, Gate
from scambot.core.locales import Locales
from scambot.core.redis import hit_rate, rate_key
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults
from tests.fakes import InMemorySettingsStore

NOW = dt.datetime(2026, 11, 13, 10, 15, tzinfo=dt.UTC)


def test_the_key_names_the_user_and_the_utc_hour() -> None:
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))

    assert rate_key("u", NOW) == "rl:u:2026111310"
    assert rate_key("u", NOW.astimezone(ist)) == "rl:u:2026111310"


async def test_hit_rate_counts_and_sets_an_expiry(redis: ArqRedis) -> None:
    assert await hit_rate(redis, "u", NOW, 3600) == (1, 3600)
    count, ttl = await hit_rate(redis, "u", NOW, 3600)

    assert count == 2
    assert 0 < ttl <= 3600


async def test_a_key_without_an_expiry_gets_one(redis: ArqRedis) -> None:
    await redis.set(rate_key("u", NOW), 4)

    assert await hit_rate(redis, "u", NOW, 3600) == (5, 3600)


async def test_a_new_hour_starts_a_new_count(redis: ArqRedis) -> None:
    await hit_rate(redis, "u", NOW, 3600)

    assert (await hit_rate(redis, "u", NOW + dt.timedelta(hours=1), 3600))[0] == 1


@pytest.fixture
def clock() -> list[dt.datetime]:
    return [NOW]


@pytest.fixture
def settings_store() -> InMemorySettingsStore:
    return InMemorySettingsStore()


@pytest.fixture
def gate(redis: ArqRedis, settings_store: InMemorySettingsStore, clock: list[dt.datetime]) -> Gate:
    runtime = RuntimeConfig(settings_store, redis, RuntimeDefaults(20, 7), 60)
    return Gate(runtime, redis, 3600, clock=lambda: clock[0])


async def test_twenty_are_allowed_and_the_next_is_limited(gate: Gate) -> None:
    for _ in range(20):
        assert (await gate.admit("u")).admission is Admission.ALLOWED

    decision = await gate.admit("u")

    assert decision.admission is Admission.LIMITED
    assert decision.limit == 20
    assert 1 <= decision.minutes <= 60


async def test_users_are_counted_separately(gate: Gate) -> None:
    for _ in range(21):
        await gate.admit("u")

    assert (await gate.admit("other")).admission is Admission.ALLOWED


async def test_a_new_hour_resets_the_limit(gate: Gate, clock: list[dt.datetime]) -> None:
    for _ in range(21):
        await gate.admit("u")
    clock[0] = NOW + dt.timedelta(hours=1)

    assert (await gate.admit("u")).admission is Admission.ALLOWED


async def test_a_blocked_user_is_refused_without_counting(
    gate: Gate, settings_store: InMemorySettingsStore, redis: ArqRedis
) -> None:
    settings_store.blocked.add("u")

    assert (await gate.admit("u")).admission is Admission.BLOCKED
    assert await redis.exists(rate_key("u", NOW)) == 0


async def test_the_limit_comes_from_the_settings_table(
    gate: Gate, settings_store: InMemorySettingsStore
) -> None:
    await settings_store.set("rate_limit_per_hour", 2)

    assert [(await gate.admit("u")).admission for _ in range(3)] == [
        Admission.ALLOWED,
        Admission.ALLOWED,
        Admission.LIMITED,
    ]


async def test_a_redis_failure_lets_the_check_through(
    settings_store: InMemorySettingsStore, redis: ArqRedis
) -> None:
    class Broken:
        def pipeline(self, **_kwargs: object) -> None:
            raise ConnectionError

    runtime = RuntimeConfig(settings_store, redis, RuntimeDefaults(20, 7), 60)
    gate = Gate(runtime, Broken(), 3600)  # type: ignore[arg-type]

    assert (await gate.admit("u")).admission is Admission.ALLOWED


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_the_message_shows_the_limit_and_the_wait(locales: Locales, lang: str) -> None:
    text = locales.get(lang).rate_limited.format(limit=20, minutes=12)

    assert "20" in text
    assert "12" in text
