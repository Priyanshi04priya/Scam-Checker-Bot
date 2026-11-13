"""A message goes through intake, the queue and the worker, using in-memory Redis."""

import logging
from typing import Any

import pytest
from arq.connections import ArqRedis
from arq.worker import create_worker

from scambot.api.intake import Intake, Outcome
from scambot.core.locales import Locales
from scambot.worker.pipeline import DEPS_KEY, PipelineDeps
from scambot.worker.settings import WorkerSettings
from tests.conftest import PipelineDepsFactory
from tests.fakes import FakeAdapter, fake_update


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-memory Redis has no INFO command, which the worker only uses for a startup log."""

    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


async def run_worker(redis: ArqRedis, deps: PipelineDeps) -> int:
    """Process every queued job once with the production worker settings, then stop."""
    worker = create_worker(
        WorkerSettings,
        redis_pool=redis,
        burst=True,
        poll_delay=0,
        handle_signals=False,
        on_startup=None,
        cron_jobs=None,
        on_shutdown=None,
        ctx={DEPS_KEY: deps},
    )
    await worker.main()
    assert worker.jobs_failed == 0
    return worker.jobs_complete


async def test_text_is_acknowledged_then_edited_into_the_verdict(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> None:
    intake = Intake(adapter, redis, locales)

    assert (
        await intake.handle(fake_update(1, "Your KYC expires today, click here"))
        is Outcome.ENQUEUED
    )
    assert adapter.log == [("ack", "7", "Checking…")]

    assert await run_worker(redis, pipeline_deps()) == 1

    en = locales.get("en")
    assert [action for action, _, _ in adapter.log] == ["ack", "edit"]
    (only_message,) = adapter.messages.values()
    assert only_message.startswith(f"🔴 {en.verdict.bands.high} · {en.verdict.labels.likely_scam}")
    assert en.verdict.signals.kyc_update in only_message
    assert only_message.endswith(en.disclaimer)


async def test_duplicate_update_gives_one_reply(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> None:
    intake = Intake(adapter, redis, locales)
    update = fake_update(2, "Forwarded as received")

    assert await intake.handle(update) is Outcome.ENQUEUED
    assert await intake.handle(update) is Outcome.DUPLICATE
    deps = pipeline_deps()
    assert await run_worker(redis, deps) == 1
    assert await intake.handle(update) is Outcome.DUPLICATE
    assert await run_worker(redis, deps) == 0

    assert [action for action, _, _ in adapter.log] == ["ack", "edit"]
    assert len(adapter.messages) == 1


async def test_hindi_user_gets_hindi_ack_and_verdict(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> None:
    intake = Intake(adapter, redis, locales)

    await intake.handle(fake_update(3, "Sach hai kya?", lang="hi"))
    await run_worker(redis, pipeline_deps())

    hi = locales.get("hi")
    assert adapter.log[0] == ("ack", "7", hi.checking)
    (only_message,) = adapter.messages.values()
    assert only_message.startswith(f"🟢 {hi.verdict.bands.low} · {hi.verdict.labels.no_issues}")
    assert only_message.endswith(hi.disclaimer)


async def test_queue_logs_never_show_the_message(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="arq")
    intake = Intake(adapter, redis, locales)

    await intake.handle(fake_update(4, "PRIVATE-MESSAGE-BODY", user_id=987654))
    await run_worker(redis, pipeline_deps())

    assert "process_job" in caplog.text
    assert "PRIVATE-MESSAGE-BODY" not in caplog.text
    assert "987654" not in caplog.text
