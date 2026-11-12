"""Redis connection, key names, update de-duplication and the job queue."""

import enum
from dataclasses import replace
from typing import Final
from urllib.parse import unquote

from arq.connections import ArqRedis, RedisSettings, create_pool
from pydantic import SecretStr

from scambot.core.contracts import Job

PROCESS_JOB: Final = "process_job"
DEDUPE_TTL_SECONDS: Final = 86_400
# How long a claim lasts before the update is marked done. It only has to outlive one attempt
# at handling the update; if the process dies mid-attempt, a redelivery after this is handled.
CLAIM_TTL_SECONDS: Final = 120

_IN_PROGRESS: Final = b"in_progress"
_DONE: Final = b"done"


class Claim(enum.Enum):
    """State of an incoming update when a caller tries to claim it."""

    NEW = "new"
    IN_PROGRESS = "in_progress"
    DONE = "done"


def dedupe_key(update_id: str) -> str:
    """Key that marks an incoming update as already handled."""
    return f"dedupe:{update_id}"


def connection_settings(redis_url: SecretStr) -> RedisSettings:
    """Queue connection settings parsed from a ``redis://`` or ``rediss://`` URL.

    arq keeps percent-escapes in the username and password, so they are decoded here the same
    way redis-py decodes them (``p%40ss`` → ``p@ss``).
    """
    settings = RedisSettings.from_dsn(redis_url.get_secret_value())
    return replace(
        settings,
        username=unquote(settings.username) if settings.username else settings.username,
        password=unquote(settings.password) if settings.password else settings.password,
    )


async def create_redis(redis_url: SecretStr) -> ArqRedis:
    """Open a Redis client that also enqueues jobs. Close it with ``await redis.aclose()``."""
    return await create_pool(connection_settings(redis_url))


async def claim_update(redis: ArqRedis, update_id: str) -> Claim:
    """Try to claim ``update_id`` for handling.

    ``NEW`` means the caller now holds a short-lived claim and must end it with
    ``finish_update`` or ``release_update``. ``IN_PROGRESS`` means another attempt holds it (or
    has just released it), so the caller should ask for a redelivery later. ``DONE`` means the
    update was handled in the last day.
    """
    key = dedupe_key(update_id)
    if await redis.set(key, _IN_PROGRESS, nx=True, ex=CLAIM_TTL_SECONDS):
        return Claim.NEW
    state = await redis.get(key)
    return Claim.DONE if state == _DONE else Claim.IN_PROGRESS


async def finish_update(redis: ArqRedis, update_id: str) -> None:
    """Mark a claimed ``update_id`` as handled, so redeliveries in the next day are dropped."""
    await redis.set(dedupe_key(update_id), _DONE, ex=DEDUPE_TTL_SECONDS)


async def release_update(redis: ArqRedis, update_id: str) -> None:
    """Drop the claim on ``update_id`` so a redelivery of the same update is handled again."""
    await redis.delete(dedupe_key(update_id))


async def enqueue(redis: ArqRedis, job: Job) -> bool:
    """Queue ``job`` for the worker; return False if a job with the same ID is already queued."""
    queued = await redis.enqueue_job(PROCESS_JOB, job, _job_id=job.job_id)
    return queued is not None
