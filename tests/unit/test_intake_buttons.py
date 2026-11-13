"""Intake: /language, /history, the verdict buttons and their answers."""

import uuid

import pytest
from arq.connections import ArqRedis
from pydantic import SecretStr

from scambot.api.gate import Gate
from scambot.api.intake import Intake, Outcome
from scambot.core.contracts import CheckerResult, ExecutedCheck, Verdict
from scambot.core.locales import Locales
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER
from scambot.core.store import NewRequest, StoredResult
from tests.fakes import (
    FakeAdapter,
    InMemorySettingsStore,
    InMemoryStore,
    fake_callback,
    fake_update,
)

ME = user_hash(7, SecretStr(DEV_USER_ID_PEPPER))


@pytest.fixture
def intake(adapter: FakeAdapter, redis: ArqRedis, locales: Locales, store: InMemoryStore) -> Intake:
    return Intake(adapter, redis, locales, users=store)


async def verdict_for(store: InMemoryStore, job_id: str = "t:1", user: str = ME) -> uuid.UUID:
    started = await store.start(NewRequest(job_id, user, "telegram", "text", "a" * 64, "en"))
    check = CheckerResult(
        checker="analyze_scam_patterns", status="ok", signal="scam", confidence=0.9, duration_ms=1
    )
    await store.save(
        started.id,
        StoredResult(
            [ExecutedCheck(result=check, input_hash="h" * 64)],
            Verdict(label="likely_scam", band="high", score=90),
            "reply",
            "en",
            [],
            "en",
            0,  # type: ignore[arg-type]
        ),
    )
    await store.set_status(started.id, "replied")
    return started.id


async def test_language_prompt_has_a_keyboard_with_short_data(
    intake: Intake, adapter: FakeAdapter, locales: Locales
) -> None:
    assert await intake.handle(fake_update(1, "/language")) is Outcome.COMMAND

    (message_id,) = adapter.keyboards
    (row,) = adapter.keyboards[message_id]
    assert [b.data for b in row] == ["l:en", "l:hi", "l:hi-Latn"]
    assert all(len(b.data.encode()) <= 64 for b in row)
    assert adapter.messages[message_id] == locales.get("en").language.prompt


async def test_language_with_an_argument_sets_it_directly(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore, locales: Locales
) -> None:
    await intake.handle(fake_update(1, "/language hi"))
    assert store.languages[ME] == "hi"
    assert adapter.log[-1][2] == locales.get("hi").language.saved.format(language="हिन्दी")

    await intake.handle(fake_update(2, "/language klingon"))
    assert adapter.log[-1][2] == locales.get("hi").language.invalid

    await intake.handle(fake_callback(3, "l:en"))
    assert store.languages[ME] == "en"
    assert adapter.log[-1][2] == locales.get("en").language.saved.format(language="English")
    assert adapter.answers == [None]


async def test_unknown_language_callbacks_are_ignored(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore
) -> None:
    await intake.handle(fake_callback(1, "l:xx"))

    assert store.languages == {}
    assert adapter.log == []
    assert adapter.answers == [None]


async def test_a_press_on_an_old_message_is_answered_in_the_users_language(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore, locales: Locales
) -> None:
    store.languages[ME] = "hi"

    await intake.handle(fake_callback(1, "x:-"))

    assert adapter.answers == [locales.get("hi").details.expired]
    assert adapter.log == []


async def test_stored_language_reaches_the_job_and_replies(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore, redis: ArqRedis, locales: Locales
) -> None:
    store.languages[ME] = "hi"

    await intake.handle(fake_update(1, "your KYC expires, share OTP", lang="en"))

    assert adapter.log == [("ack", "7", locales.get("hi").checking)]
    await intake.handle(fake_update(2, "/help", lang="en"))
    assert adapter.log[-1][2] == locales.get("hi").commands.help
    (queued,) = await redis.queued_jobs()
    assert queued.args[0].user_lang == "hi"


async def test_details_feedback_and_ownership(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore, locales: Locales
) -> None:
    request_id = await verdict_for(store)
    en = locales.get("en")

    assert await intake.handle(fake_callback(1, f"d:{request_id}")) is Outcome.CALLBACK
    assert en.details.title in adapter.log[-1][2]
    await intake.handle(fake_callback(2, f"h:{request_id}"))
    await intake.handle(fake_callback(3, f"w:{request_id}"))
    await intake.handle(fake_callback(4, f"w:{request_id}", user_id=8))
    await intake.handle(fake_callback(5, f"h:{uuid.uuid4()}"))
    await intake.handle(fake_callback(6, "h:not-a-uuid"))

    assert adapter.answers == [
        None,
        en.feedback.thanks_helpful,
        en.feedback.thanks_wrong,
        en.feedback.unavailable,
        en.feedback.unavailable,
        en.feedback.unavailable,
    ]
    assert store.feedback == {(request_id, ME): False}


async def test_a_duplicate_press_is_dropped(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore
) -> None:
    request_id = await verdict_for(store)

    assert await intake.handle(fake_callback(1, f"h:{request_id}")) is Outcome.CALLBACK
    assert await intake.handle(fake_callback(1, f"h:{request_id}")) is Outcome.DUPLICATE

    assert len(adapter.answers) == 1


async def test_the_press_is_answered_even_when_the_store_fails(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore, locales: Locales
) -> None:
    request_id = await verdict_for(store)
    store.fail = True

    assert await intake.handle(fake_callback(1, f"d:{request_id}")) is Outcome.CALLBACK

    assert adapter.answers == [locales.get("en").feedback.unavailable]


async def test_a_failing_answer_does_not_fail_the_update(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore
) -> None:
    request_id = await verdict_for(store)
    adapter.fail_answer = True

    assert await intake.handle(fake_callback(1, f"h:{request_id}")) is Outcome.CALLBACK


async def test_a_blocked_user_gets_no_reply_but_the_press_is_answered(
    adapter: FakeAdapter, redis: ArqRedis, locales: Locales, store: InMemoryStore
) -> None:
    request_id = await verdict_for(store)
    settings_store = InMemorySettingsStore(blocked={ME})
    runtime = RuntimeConfig(settings_store, redis, RuntimeDefaults(20, 7), 60)
    intake = Intake(adapter, redis, locales, gate=Gate(runtime, redis, 3600), users=store)

    await intake.handle(fake_callback(1, f"d:{request_id}"))

    assert adapter.log == []
    assert adapter.answers == [None]
    assert store.feedback == {}


async def test_history_empty_and_populated(
    intake: Intake, adapter: FakeAdapter, store: InMemoryStore, locales: Locales
) -> None:
    en = locales.get("en")
    await intake.handle(fake_update(1, "/history"))
    assert adapter.log[-1][2] == en.history.empty

    await verdict_for(store, "t:1")
    await verdict_for(store, "t:2")
    await intake.handle(fake_update(2, "/history"))

    lines = adapter.log[-1][2].splitlines()
    assert lines[0] == en.history.title
    assert len(lines) == 3
    assert en.verdict.labels.likely_scam in lines[1]
