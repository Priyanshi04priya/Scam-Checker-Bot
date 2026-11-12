"""A message goes through intake, the queue and the worker, using in-memory Redis."""

import logging
from typing import Any

import pytest
from arq.connections import ArqRedis
from arq.worker import create_worker

from scambot.api.intake import Intake, Outcome
from scambot.core.locales import Locales
from scambot.worker.pipeline import ADAPTERS_KEY, LOCALES_KEY
from scambot.worker.settings import WorkerSettings
from tests.fakes import FakeAdapter, fake_update


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-memory Redis has no INFO command, which the worker only uses for a startup log."""

    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


async def run_worker(redis: ArqRedis, adapter: FakeAdapter, locales: Locales) -> int:
    """Process every queued job once with the production worker settings, then stop."""
    worker = create_worker(
        WorkerSettings,
        redis_pool=redis,
        burst=True,
        poll_delay=0,
        handle_signals=False,
        on_startup=None,
        on_shutdown=None,
        ctx={ADAPTERS_KEY: {"telegram": adapter}, LOCALES_KEY: locales},
    )
    await worker.main()
    assert worker.jobs_failed == 0
    return worker.jobs_complete


async def test_text_is_acknowledged_then_edited_into_the_verdict(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales
) -> None:
    intake = Intake(adapter, redis, locales)

    assert (
        await intake.handle(fake_update(1, "Your KYC expires today, click here"))
        is Outcome.ENQUEUED
    )
    assert adapter.log == [("ack", "7", "Checking…")]

    assert await run_worker(redis, adapter, locales) == 1

    en = locales.get("en")
    assert [action for action, _, _ in adapter.log] == ["ack", "edit"]
    (only_message,) = adapter.messages.values()
    assert only_message.startswith(
        f"{en.verdict.bands.unknown} · {en.verdict.labels.could_not_verify}"
    )
    assert en.disclaimer in only_message


async def test_duplicate_update_gives_one_reply(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales
) -> None:
    intake = Intake(adapter, redis, locales)
    update = fake_update(2, "Forwarded as received")

    assert await intake.handle(update) is Outcome.ENQUEUED
    assert await intake.handle(update) is Outcome.DUPLICATE
    assert await run_worker(redis, adapter, locales) == 1
    assert await intake.handle(update) is Outcome.DUPLICATE
    assert await run_worker(redis, adapter, locales) == 0

    assert [action for action, _, _ in adapter.log] == ["ack", "edit"]
    assert len(adapter.messages) == 1


async def test_hindi_user_gets_hindi_ack_and_verdict(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales
) -> None:
    intake = Intake(adapter, redis, locales)

    await intake.handle(fake_update(3, "Sach hai kya?", lang="hi"))
    await run_worker(redis, adapter, locales)

    hi = locales.get("hi")
    assert adapter.log[0] == ("ack", "7", hi.checking)
    (only_message,) = adapter.messages.values()
    assert hi.verdict.labels.could_not_verify in only_message


async def test_queue_logs_never_show_the_message(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="arq")
    intake = Intake(adapter, redis, locales)

    await intake.handle(fake_update(4, "PRIVATE-MESSAGE-BODY", user_id=987654))
    await run_worker(redis, adapter, locales)

    assert "process_job" in caplog.text
    assert "PRIVATE-MESSAGE-BODY" not in caplog.text
    assert "987654" not in caplog.text
