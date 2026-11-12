"""Tests for the job pipeline and the worker settings."""

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from aiogram import Bot

from scambot.agent.analyze import Analysis
from scambot.core.config import AppConfig
from scambot.core.contracts import Job
from scambot.core.llm import DisabledLLMClient, PromptLibrary
from scambot.core.locales import Locales
from scambot.core.redis import PROCESS_JOB
from scambot.mcp_server.checkers.scam import ScamClassification
from scambot.preprocess.entities import HeadRedirectExpander, NoopExpander
from scambot.worker import settings as worker_settings
from scambot.worker.cron import INGEST_KEY
from scambot.worker.pipeline import DEPS_KEY, PipelineDeps, process_job
from scambot.worker.settings import WorkerSettings
from tests.conftest import PipelineDepsFactory
from tests.fakes import FailingToolCaller, FakeAdapter, FakeLLMClient, InMemoryStore
from tests.samples import ELECTRICITY_EN, ORDINARY


def a_job(text: str = ELECTRICITY_EN, lang: str | None = "en", ack: str | None = None) -> Job:
    return Job(
        job_id="telegram:1",
        trace_id="trace-1",
        channel="telegram",
        chat_id="7",
        user_hash="h",
        kind="text",
        text=text,
        ack_message_id=ack,
        preferred_lang=lang,
        received_at=datetime.now(UTC),
    )


def ctx(deps: PipelineDeps) -> dict[str, Any]:
    return {DEPS_KEY: deps}


def only_reply(adapter: FakeAdapter) -> str:
    (message,) = adapter.messages.values()
    return message


async def test_scam_text_is_answered_and_stored(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, store: InMemoryStore, locales: Locales
) -> None:
    await process_job(ctx(pipeline_deps()), a_job())

    en = locales.get("en")
    assert only_reply(adapter).startswith(
        f"🔴 {en.verdict.bands.high} · {en.verdict.labels.likely_scam}"
    )
    (row,) = store.requests.values()
    assert row.status == "replied"
    assert row.replied_at is not None
    assert row.verdict is not None
    assert row.verdict.label == "likely_scam"
    assert [c.result.checker for c in row.checks] == ["analyze_scam_patterns"]
    assert row.request.sha256 is not None
    assert row.reply_language == "en"


async def test_replied_request_is_skipped(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, store: InMemoryStore
) -> None:
    deps = pipeline_deps()
    await process_job(ctx(deps), a_job())
    await process_job(ctx(deps), a_job())

    assert len(adapter.log) == 1


async def test_store_failure_still_replies(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter
) -> None:
    await process_job(ctx(pipeline_deps(request_store=InMemoryStore(fail=True))), a_job())

    assert only_reply(adapter).startswith("🔴")


async def test_analysis_failure_uses_the_baseline_plan(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, fake_llm: FakeLLMClient
) -> None:
    await process_job(ctx(pipeline_deps()), a_job(ORDINARY[0]))

    assert only_reply(adapter).startswith("🟢")
    assert fake_llm.names()[0] == "analyze"


async def test_analysis_language_and_claims_are_used(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    fake_llm: FakeLLMClient,
    store: InMemoryStore,
    locales: Locales,
) -> None:
    fake_llm.responses["analyze"] = Analysis(
        claims=["Power will be cut tonight"],
        reply_language="hi",
        claimed_date=None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )

    await process_job(ctx(pipeline_deps()), a_job(lang="en"))

    assert only_reply(adapter).startswith(f"🔴 {locales.get('hi').verdict.bands.high}")
    (row,) = store.requests.values()
    assert row.claims == ["Power will be cut tonight"]
    assert row.language == "hi"


def tiny_deadline(app_config: AppConfig, deadline_s: float) -> AppConfig:
    pipeline = app_config.thresholds.pipeline.model_copy(update={"deadline_s": deadline_s})
    thresholds = app_config.thresholds.model_copy(update={"pipeline": pipeline})
    return dataclasses.replace(app_config, thresholds=thresholds)


class SlowExplanation(FakeLLMClient):
    async def structured(self, prompt_name: str, *args: Any, **kwargs: Any) -> Any:
        if prompt_name == "explanation":
            await asyncio.sleep(5)
        return await super().structured(prompt_name, *args, **kwargs)


async def test_deadline_with_partial_results(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    store: InMemoryStore,
    app_config: AppConfig,
    prompts: PromptLibrary,
) -> None:
    deps = pipeline_deps(llm=SlowExplanation(prompts), config=tiny_deadline(app_config, 0.5))

    await process_job(ctx(deps), a_job())

    assert only_reply(adapter).startswith("🔴")
    (row,) = store.requests.values()
    assert row.status == "timeout"
    assert row.verdict is not None


async def test_deadline_with_nothing_finished(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    store: InMemoryStore,
    app_config: AppConfig,
    locales: Locales,
) -> None:
    deps = pipeline_deps(config=tiny_deadline(app_config, 0.000001))

    await process_job(ctx(deps), a_job(lang="hi"))

    assert only_reply(adapter) == locales.get("hi").timeout
    (row,) = store.requests.values()
    assert row.status == "timeout"
    assert row.verdict is None


async def test_unexpected_error_marks_failed(
    pipeline_deps: PipelineDepsFactory, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: Any) -> Any:
        msg = "bug"
        raise RuntimeError(msg)

    monkeypatch.setattr("scambot.worker.pipeline.build_plan", explode)

    with pytest.raises(RuntimeError):
        await process_job(ctx(pipeline_deps()), a_job())

    (row,) = store.requests.values()
    assert row.status == "failed"


async def test_checker_server_down_gives_could_not_verify(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, locales: Locales
) -> None:
    await process_job(ctx(pipeline_deps(caller=FailingToolCaller())), a_job())

    en = locales.get("en").verdict
    reply = only_reply(adapter)
    assert reply.startswith(f"⚪ {en.bands.unknown} · {en.labels.could_not_verify}")
    assert en.unavailable.replace("{checks}", en.checkers.analyze_scam_patterns) in reply


async def test_cost_counts_both_usage_sources(
    pipeline_deps: PipelineDepsFactory, fake_llm: FakeLLMClient, store: InMemoryStore
) -> None:
    fake_llm.responses["scam_classify"] = ScamClassification(
        template="none", spans=[], confidence=0.0
    )

    await process_job(ctx(pipeline_deps()), a_job())

    # analyze + explanation recorded in the worker, scam_classify inside the checker; each
    # fake call is 100 input + 20 output tokens at 4 and 20 dollars per million.
    assert sorted(fake_llm.names()) == ["analyze", "explanation", "scam_classify"]
    (row,) = store.requests.values()
    assert row.cost_usd == Decimal("0.002400")


def test_worker_settings() -> None:
    assert WorkerSettings.job_timeout == 100
    assert WorkerSettings.max_tries == 3
    assert WorkerSettings.keep_result == 0
    assert WorkerSettings.max_jobs == 10
    assert [getattr(f, "name", None) for f in WorkerSettings.functions] == [PROCESS_JOB]
    assert WorkerSettings.cron_jobs is not None
    (job,) = WorkerSettings.cron_jobs
    assert job.name == "ingest_factchecks"
    assert (job.hour, job.minute) == ({2}, {0})
    assert job.unique is True
    assert job.timeout_s == 1800
    assert job.max_tries == 1
    assert job.run_at_startup is False
    assert WorkerSettings.timezone.utcoffset(None) == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("expand", ["true", "false"])
async def test_worker_startup_and_shutdown(monkeypatch: pytest.MonkeyPatch, expand: str) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:" + "A" * 35)
    monkeypatch.setenv("EXPAND_SHORT_URLS", expand)
    context: dict[str, Any] = {}

    await worker_settings.startup(context)

    assert isinstance(context[worker_settings.BOT_KEY], Bot)
    deps: PipelineDeps = context[DEPS_KEY]
    assert set(deps.adapters) == {"telegram"}
    assert isinstance(deps.llm, DisabledLLMClient)
    expander_type = HeadRedirectExpander if expand == "true" else NoopExpander
    assert isinstance(deps.expander, expander_type)
    assert context[INGEST_KEY] is None  # no embedding model installed in the test environment
    await worker_settings.shutdown(context)
    await worker_settings.shutdown({})


async def test_worker_startup_with_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:" + "A" * 35)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    context: dict[str, Any] = {}

    await worker_settings.startup(context)

    assert context[worker_settings.ANTHROPIC_KEY] is not None
    await worker_settings.shutdown(context)
