"""Tests for the stub pipeline and the worker settings."""

from datetime import UTC, datetime
from typing import Any

import pytest
from aiogram import Bot

from scambot.core.contracts import Job
from scambot.core.locales import Locales
from scambot.core.redis import PROCESS_JOB
from scambot.worker import settings as worker_settings
from scambot.worker.pipeline import (
    ADAPTERS_KEY,
    LOCALES_KEY,
    STUB_VERDICT,
    process_job,
    render_reply,
)
from scambot.worker.settings import WorkerSettings
from tests.fakes import FakeAdapter


def a_job(ack_message_id: str | None, lang: str | None = "en") -> Job:
    return Job(
        job_id="telegram:1",
        trace_id="trace-1",
        channel="telegram",
        chat_id="7",
        user_hash="h",
        kind="text",
        text="x",
        ack_message_id=ack_message_id,
        preferred_lang=lang,
        received_at=datetime.now(UTC),
    )


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_render_reply(locales: Locales, lang: str) -> None:
    texts = locales.get(lang)

    reply = render_reply(STUB_VERDICT, locales, lang)

    assert reply.splitlines()[0] == (
        f"{texts.verdict.bands.unknown} · {texts.verdict.labels.could_not_verify}"
    )
    assert texts.verdict.stub_note in reply
    assert reply.endswith(texts.disclaimer)


def test_stub_verdict_claims_nothing() -> None:
    assert STUB_VERDICT.label == "could_not_verify"
    assert STUB_VERDICT.band == "unknown"
    assert STUB_VERDICT.source_confirmed is False


async def test_process_job_edits_the_ack(adapter: FakeAdapter, locales: Locales) -> None:
    ack_id = await adapter.send_ack(a_job(None), "Checking…")
    ctx: dict[str, Any] = {ADAPTERS_KEY: {"telegram": adapter}, LOCALES_KEY: locales}

    await process_job(ctx, a_job(ack_id))
    await process_job(ctx, a_job(ack_id))  # a retried job edits the same message again

    assert adapter.messages == {ack_id: render_reply(STUB_VERDICT, locales, "en")}


def test_worker_settings() -> None:
    assert WorkerSettings.job_timeout == 100
    assert WorkerSettings.max_tries == 3
    assert WorkerSettings.keep_result == 0
    assert WorkerSettings.max_jobs == 10
    assert [getattr(f, "name", None) for f in WorkerSettings.functions] == [PROCESS_JOB]


async def test_worker_startup_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:" + "A" * 35)
    ctx: dict[str, Any] = {}

    await worker_settings.startup(ctx)

    assert isinstance(ctx[worker_settings.BOT_KEY], Bot)
    assert set(ctx[ADAPTERS_KEY]) == {"telegram"}
    assert isinstance(ctx[LOCALES_KEY], Locales)
    await worker_settings.shutdown(ctx)
    await worker_settings.shutdown({})
