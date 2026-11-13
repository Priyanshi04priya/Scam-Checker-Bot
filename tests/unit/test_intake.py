"""Tests for intake: commands, de-duplication, acknowledgement and enqueueing."""

import datetime as dt

import pytest
from arq.connections import ArqRedis
from pydantic import SecretStr
from redis.exceptions import ConnectionError as RedisConnectionError

from scambot.api import intake as intake_module
from scambot.api.gate import Gate
from scambot.api.intake import Intake, Outcome
from scambot.core.locales import Locales
from scambot.core.redis import CLAIM_TTL_SECONDS, claim_update, dedupe_key
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER
from tests.fakes import (
    FakeAdapter,
    InMemorySettingsStore,
    InMemoryStore,
    fake_media_update,
    fake_update,
)
from tests.media import png_bytes


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


async def test_rejection_is_answered_in_the_users_language(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis, locales: Locales
) -> None:
    update = {"update_id": 11, "user_id": 7, "lang": "hi", "reject": "unsupported_type"}

    assert await intake.handle(update) is Outcome.REJECTED
    assert await intake.handle(update) is Outcome.DUPLICATE

    assert adapter.log == [("text", "7", locales.get("hi").rejected.unsupported_type)]
    assert await redis.queued_jobs() == []


async def test_declared_oversized_voice_is_rejected_without_queueing(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis, locales: Locales
) -> None:
    update = fake_media_update(12, "voice", "v", size=25 * 1024 * 1024, source="voice")

    assert await intake.handle(update) is Outcome.REJECTED

    assert adapter.log == [("text", "7", locales.get("en").rejected.too_large)]
    assert await redis.queued_jobs() == []


async def test_declared_wrong_type_is_rejected(intake: Intake, adapter: FakeAdapter) -> None:
    update = fake_media_update(13, "image", "i", mime="application/pdf", source="document")

    assert await intake.handle(update) is Outcome.REJECTED
    assert adapter.log[0][0] == "text"


async def test_valid_media_job_is_enqueued_with_its_metadata(
    intake: Intake, adapter: FakeAdapter, redis: ArqRedis
) -> None:
    adapter.files["i"] = png_bytes()
    update = fake_media_update(14, "image", "i", caption="look", size=1000, mime="image/png")

    assert await intake.handle(update) is Outcome.ENQUEUED

    (queued,) = await redis.queued_jobs()
    (job,) = queued.args
    assert job.kind == "image"
    assert job.media_ref == "i"
    assert job.media is not None
    assert (job.media.file_size, job.media.mime_type) == (1000, "image/png")
    assert adapter.log == [("ack", "7", "Checking…")]


# --- limits, blocking and /delete ----------------------------------------------------------

NOW = dt.datetime(2026, 11, 13, 10, 15, tzinfo=dt.UTC)


class Users(InMemoryStore):
    """Records /delete calls; ``fail`` makes them raise."""

    def __init__(self, *, fail: bool = False, removed: int = 3) -> None:
        super().__init__()
        self.fail = fail
        self.removed = removed
        self.calls: list[str] = []

    async def delete_user(self, user_hash: str) -> int:
        self.calls.append(user_hash)
        if self.fail:
            msg = "database unavailable"
            raise ConnectionError(msg)
        return self.removed


@pytest.fixture
def clock() -> list[dt.datetime]:
    return [NOW]


@pytest.fixture
def settings_store() -> InMemorySettingsStore:
    return InMemorySettingsStore()


@pytest.fixture
def runtime(redis: ArqRedis, settings_store: InMemorySettingsStore) -> RuntimeConfig:
    return RuntimeConfig(settings_store, redis, RuntimeDefaults(20, 7), 60)


@pytest.fixture
def users() -> Users:
    return Users()


@pytest.fixture
def guarded(
    adapter: FakeAdapter,
    redis: ArqRedis,
    locales: Locales,
    runtime: RuntimeConfig,
    clock: list[dt.datetime],
    users: Users,
) -> Intake:
    gate = Gate(runtime, redis, 3600, clock=lambda: clock[0])
    return Intake(adapter, redis, locales, gate=gate, users=users)


@pytest.mark.parametrize("lang", ["en", "hi"])
async def test_the_21st_check_in_an_hour_is_refused(
    guarded: Intake, adapter: FakeAdapter, redis: ArqRedis, locales: Locales, lang: str
) -> None:
    for i in range(20):
        assert await guarded.handle(fake_update(i, "text", lang=lang)) is Outcome.ENQUEUED
    adapter.log.clear()

    assert await guarded.handle(fake_update(99, "text", lang=lang)) is Outcome.RATE_LIMITED

    text = adapter.log[0][2]
    assert adapter.log == [("text", "7", text)]  # no "Checking…"
    assert "20" in text
    assert text.startswith(locales.get(lang).rate_limited.split("{")[0])
    assert len(await redis.queued_jobs()) == 20
    assert await redis.exists(dedupe_key("telegram:99")) == 1  # handled, not redelivered


async def test_commands_and_rejections_are_not_counted(
    guarded: Intake, redis: ArqRedis, adapter: FakeAdapter
) -> None:
    for i in range(30):
        await guarded.handle(fake_update(i, "/help"))
    await guarded.handle(
        fake_media_update(100, "image", "f", size=10**9, mime="image/png", source="photo")
    )

    assert await guarded.handle(fake_update(200, "text")) is Outcome.ENQUEUED
    keys = [key.decode() async for key in redis.scan_iter(match="rl:*")]
    assert keys == [f"rl:{hashed('7')}:2026111310"]


def hashed(user_id: str) -> str:
    return user_hash(int(user_id), SecretStr(DEV_USER_ID_PEPPER))


async def test_a_failed_enqueue_gives_the_charge_back(
    guarded: Intake, redis: ArqRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*_args: object, **_kwargs: object) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr("scambot.api.intake.enqueue", broken)
    with pytest.raises(ConnectionError):
        await guarded.handle(fake_update(1, "text"))

    assert [key async for key in redis.scan_iter(match="rl:*")] == []


async def test_a_new_hour_allows_checks_again(guarded: Intake, clock: list[dt.datetime]) -> None:
    for i in range(21):
        await guarded.handle(fake_update(i, "text"))
    clock[0] = NOW + dt.timedelta(hours=1)

    assert await guarded.handle(fake_update(500, "text")) is Outcome.ENQUEUED


async def test_the_limit_can_be_changed_in_the_settings_table(
    guarded: Intake, settings_store: InMemorySettingsStore, runtime: RuntimeConfig
) -> None:
    await settings_store.set("rate_limit_per_hour", 1)
    await runtime.invalidate("rate_limit_per_hour")

    assert await guarded.handle(fake_update(1, "text")) is Outcome.ENQUEUED
    assert await guarded.handle(fake_update(2, "text")) is Outcome.RATE_LIMITED


async def test_a_blocked_user_is_ignored_silently(
    guarded: Intake,
    adapter: FakeAdapter,
    redis: ArqRedis,
    settings_store: InMemorySettingsStore,
    runtime: RuntimeConfig,
) -> None:
    settings_store.blocked.add(hashed("7"))

    assert await guarded.handle(fake_update(1, "text")) is Outcome.BLOCKED

    assert adapter.log == []
    assert await redis.queued_jobs() == []
    # Commands are still answered, so the user is not left wondering.
    assert await guarded.handle(fake_update(2, "/help")) is Outcome.COMMAND

    settings_store.blocked.clear()
    await runtime.invalidate(f"blocked:{hashed('7')}")
    assert await guarded.handle(fake_update(3, "text")) is Outcome.ENQUEUED


@pytest.mark.parametrize("lang", ["en", "hi"])
async def test_delete_removes_the_users_data_and_confirms(
    guarded: Intake,
    adapter: FakeAdapter,
    redis: ArqRedis,
    locales: Locales,
    users: Users,
    lang: str,
) -> None:
    await guarded.handle(fake_update(1, "text", lang=lang))
    adapter.log.clear()
    assert [key async for key in redis.scan_iter(match="rl:*")]

    assert await guarded.handle(fake_update(2, "/delete", lang=lang)) is Outcome.COMMAND

    assert users.calls == [hashed("7")]
    assert adapter.log == [("text", "7", locales.get(lang).commands.delete.format(count=3))]
    assert [key async for key in redis.scan_iter(match="rl:*")]  # /delete must not reset the limit


async def test_a_failed_delete_is_reported_not_claimed(
    guarded: Intake, adapter: FakeAdapter, users: Users, locales: Locales
) -> None:
    users.fail = True

    assert await guarded.handle(fake_update(2, "/delete")) is Outcome.COMMAND

    assert adapter.log == [("text", "7", locales.get("en").commands.delete_failed)]


async def test_delete_without_a_user_store_is_an_unknown_command(
    intake: Intake, adapter: FakeAdapter, locales: Locales
) -> None:
    await intake.handle(fake_update(1, "/delete"))

    assert adapter.log[0][2] == locales.get("en").commands.unknown


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_help_and_start_mention_delete(locales: Locales, lang: str) -> None:
    commands = locales.get(lang).commands

    assert "/delete" in commands.help
    assert "/delete" in commands.start
    assert "/delete" in commands.privacy
