"""Tests for intake: commands, de-duplication, acknowledgement and enqueueing."""

import pytest
from arq.connections import ArqRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from scambot.api import intake as intake_module
from scambot.api.intake import Intake, Outcome
from scambot.core.locales import Locales
from scambot.core.redis import CLAIM_TTL_SECONDS, claim_update, dedupe_key
from tests.fakes import FakeAdapter, fake_update


@pytest.fixture
def intake(adapter: FakeAdapter, redis: ArqRedis, locales: Locales) -> Intake:
    return Intake(adapter, redis, locales)


@pytest.mark.parametrize("lang", ["en", "hi"])
@pytest.mark.parametrize("command", ["start", "help", "privacy"])
async def test_commands_are_answered_in_the_users_language(
    intake: Intake, adapter: FakeAdapter, locales: Locales, command: str, lang: str
) -> None:
    outcome = await intake.handle(fake_update(1, f"/{command}", lang=lang))

    expected = getattr(locales.get(lang).commands, command)
    assert outcome is Outcome.COMMAND
    assert adapter.log == [("text", "7", expected)]


async def test_regional_language_code_uses_the_base_language(
    intake: Intake, adapter: FakeAdapter, locales: Locales
) -> None:
    await intake.handle(fake_update(1, "/help", lang="hi-IN"))

    assert adapter.log[0][2] == locales.get("hi").commands.help


async def test_unsupported_language_falls_back_to_english(
    intake: Intake, adapter: FakeAdapter, locales: Locales
) -> None:
    await intake.handle(fake_update(1, "/start", lang="ta"))

    assert adapter.log[0][2] == locales.get("en").commands.start


async def test_unknown_command_points_to_help(
    intake: Intake, adapter: FakeAdapter, locales: Locales
) -> None:
    await intake.handle(fake_update(1, "/frobnicate", lang="hi"))

    assert adapter.log[0][2] == locales.get("hi").commands.unknown


async def test_duplicate_command_is_answered_once(intake: Intake, adapter: FakeAdapter) -> None:
    assert await intake.handle(fake_update(1, "/start")) is Outcome.COMMAND
    assert await intake.handle(fake_update(1, "/start")) is Outcome.DUPLICATE

    assert len(adapter.log) == 1


async def test_text_is_acknowledged_and_enqueued(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis
) -> None:
    assert await intake.handle(fake_update(5, "Send OTP to claim prize")) is Outcome.ENQUEUED

    assert adapter.log == [("ack", "7", "Checking…")]
    (queued,) = await redis.queued_jobs()
    assert queued.function == "process_job"
    assert queued.job_id == "telegram:5"
    (job,) = queued.args
    assert job.ack_message_id == "1"
    assert job.text == "Send OTP to claim prize"
    assert await redis.ttl(dedupe_key("telegram:5")) == 86_400


async def test_duplicate_text_is_acknowledged_once(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis
) -> None:
    await intake.handle(fake_update(5, "same"))
    assert await intake.handle(fake_update(5, "same")) is Outcome.DUPLICATE

    assert len(adapter.log) == 1
    assert len(await redis.queued_jobs()) == 1


async def test_different_updates_with_same_text_are_both_checked(
    intake: Intake, redis: ArqRedis
) -> None:
    await intake.handle(fake_update(5, "same"))
    await intake.handle(fake_update(6, "same"))

    assert len(await redis.queued_jobs()) == 2


async def test_ignored_update_sends_nothing(intake: Intake, adapter: FakeAdapter) -> None:
    assert await intake.handle({"update_id": 9}) is Outcome.IGNORED
    assert adapter.log == []


async def test_failed_ack_releases_the_update_for_a_retry(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis
) -> None:
    adapter.fail_ack = True
    with pytest.raises(ConnectionError):
        await intake.handle(fake_update(5, "text"))
    assert await redis.exists(dedupe_key("telegram:5")) == 0

    adapter.fail_ack = False
    assert await intake.handle(fake_update(5, "text")) is Outcome.ENQUEUED


async def test_update_claimed_by_another_attempt_is_left_to_it(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis
) -> None:
    await claim_update(redis, "telegram:5")  # e.g. a slow first delivery, or one that crashed

    assert await intake.handle(fake_update(5, "text")) is Outcome.IN_PROGRESS
    assert adapter.log == []
    assert await redis.queued_jobs() == []
    assert 0 < await redis.ttl(dedupe_key("telegram:5")) <= CLAIM_TTL_SECONDS


async def test_failed_release_does_not_hide_the_original_error(
    intake: Intake, adapter: FakeAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def redis_down(*_args: object) -> None:
        msg = "redis unavailable"
        raise RedisConnectionError(msg)

    monkeypatch.setattr(intake_module, "release_update", redis_down)
    adapter.fail_ack = True

    with pytest.raises(ConnectionError, match="platform unavailable"):
        await intake.handle(fake_update(5, "text"))


async def test_failed_finish_still_reports_the_update_as_handled(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def redis_down(*_args: object) -> None:
        msg = "redis unavailable"
        raise RedisConnectionError(msg)

    monkeypatch.setattr(intake_module, "finish_update", redis_down)

    assert await intake.handle(fake_update(5, "text")) is Outcome.ENQUEUED
    assert len(await redis.queued_jobs()) == 1
    # The short claim still drops an immediate redelivery.
    assert await intake.handle(fake_update(5, "text")) is Outcome.IN_PROGRESS
    assert len(adapter.log) == 1
