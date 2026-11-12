"""Tests for Redis keys, update de-duplication and enqueueing."""

from datetime import UTC, datetime

from arq.connections import ArqRedis
from pydantic import SecretStr

from scambot.core.contracts import Job
from scambot.core.redis import (
    CLAIM_TTL_SECONDS,
    DEDUPE_TTL_SECONDS,
    Claim,
    claim_update,
    connection_settings,
    dedupe_key,
    enqueue,
    finish_update,
    release_update,
)


def test_dedupe_key() -> None:
    assert dedupe_key("telegram:5") == "dedupe:telegram:5"


def test_connection_settings_from_url() -> None:
    settings = connection_settings(SecretStr("redis://:pw@cache.internal:6380/2"))

    assert (settings.host, settings.port, settings.database) == ("cache.internal", 6380, 2)
    assert settings.password == "pw"


def test_connection_settings_decode_escaped_credentials() -> None:
    settings = connection_settings(SecretStr("rediss://app%3Auser:p%40ss%2Fw0rd@cache:6380/0"))

    assert (settings.username, settings.password, settings.ssl) == ("app:user", "p@ss/w0rd", True)


async def test_claim_update_only_once(redis: ArqRedis) -> None:
    assert await claim_update(redis, "telegram:1") is Claim.NEW
    assert await claim_update(redis, "telegram:1") is Claim.IN_PROGRESS
    assert await claim_update(redis, "telegram:2") is Claim.NEW


async def test_claim_is_short_lived_until_the_update_is_finished(redis: ArqRedis) -> None:
    # A process that dies mid-attempt must not block the platform's redelivery for a day.
    await claim_update(redis, "telegram:1")
    assert await redis.ttl(dedupe_key("telegram:1")) == CLAIM_TTL_SECONDS

    await finish_update(redis, "telegram:1")

    assert await redis.ttl(dedupe_key("telegram:1")) == DEDUPE_TTL_SECONDS
    assert await claim_update(redis, "telegram:1") is Claim.DONE


async def test_release_update_allows_a_new_claim(redis: ArqRedis) -> None:
    await claim_update(redis, "telegram:1")
    await release_update(redis, "telegram:1")

    assert await claim_update(redis, "telegram:1") is Claim.NEW


async def test_enqueue_uses_the_job_id_and_refuses_a_second_copy(redis: ArqRedis) -> None:
    job = Job(
        job_id="telegram:1",
        trace_id="t",
        channel="telegram",
        chat_id="1",
        user_hash="h",
        kind="text",
        text="x",
        received_at=datetime.now(UTC),
    )

    assert await enqueue(redis, job) is True
    assert await enqueue(redis, job) is False
    (queued,) = await redis.queued_jobs()
    assert (queued.function, queued.job_id, queued.args) == ("process_job", "telegram:1", (job,))
