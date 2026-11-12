"""Scam text end to end: intake, queue, worker, in-process checker server and the real checker."""

import io
import logging
from collections.abc import Mapping
from typing import Any

import pytest
from arq.connections import ArqRedis
from arq.worker import create_worker

from scambot.agent.analyze import Analysis
from scambot.agent.reply import Explanation
from scambot.api.intake import Intake, Outcome
from scambot.core.contracts import Job
from scambot.core.llm import DisabledLLMClient
from scambot.core.locales import Locales
from scambot.core.logging import configure_logging
from scambot.worker.pipeline import DEPS_KEY, PipelineDeps, process_job
from scambot.worker.settings import WorkerSettings
from tests.conftest import PipelineDepsFactory
from tests.fakes import FailingToolCaller, FakeAdapter, FakeLLMClient, InMemoryStore, fake_update
from tests.samples import ELECTRICITY_EN, ELECTRICITY_HI, ORDINARY


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


async def run_worker(redis: ArqRedis, deps: PipelineDeps) -> int:
    worker = create_worker(
        WorkerSettings,
        redis_pool=redis,
        burst=True,
        poll_delay=0,
        handle_signals=False,
        on_startup=None,
        on_shutdown=None,
        ctx={DEPS_KEY: deps},
    )
    await worker.main()
    assert worker.jobs_failed == 0
    return worker.jobs_complete


async def check(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    deps: PipelineDeps,
    update: Mapping[str, Any],
) -> str:
    intake = Intake(adapter, redis, locales)
    assert await intake.handle(update) is Outcome.ENQUEUED
    assert await run_worker(redis, deps) == 1
    assert adapter.log[-1][0] == "edit"
    return adapter.log[-1][2]


def hindi_analysis(variables: Mapping[str, str]) -> Analysis:
    return Analysis(
        claims=[],
        reply_language="hi",
        claimed_date=None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )


async def test_electricity_scam_text_gets_high_risk(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    reply = await check(redis, adapter, locales, pipeline_deps(), fake_update(1, ELECTRICITY_EN))

    en = locales.get("en")
    assert reply.startswith("🔴 High risk · Likely scam")
    for signal in ("electricity_disconnection", "urgency", "phone_number"):
        assert getattr(en.verdict.signals, signal) in reply
    assert reply.endswith(en.disclaimer)
    (row,) = store.requests.values()
    assert row.status == "replied"
    assert len(row.checks) == 1
    assert row.verdict is not None
    assert row.verdict.label == "likely_scam"


async def test_llm_explanation_with_a_link_is_replaced_by_the_template(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.responses["explanation"] = Explanation(
        explanation="This is a scam. Report it at https://cybercrime.gov.in right away."
    )

    reply = await check(redis, adapter, locales, pipeline_deps(), fake_update(2, ELECTRICITY_EN))

    assert "cybercrime" not in reply
    for phrase in (
        "an electricity disconnection threat",
        "pressure to act urgently",
        "a phone number to call",
    ):
        assert phrase in reply


async def test_ordinary_message_gets_low_risk(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> None:
    reply = await check(redis, adapter, locales, pipeline_deps(), fake_update(3, ORDINARY[0]))

    assert reply.startswith("🟢 Low risk · No issues found")


async def test_hindi_message_gets_a_hindi_reply(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
) -> None:
    fake_llm.responses["analyze"] = hindi_analysis

    reply = await check(
        redis, adapter, locales, pipeline_deps(), fake_update(4, ELECTRICITY_HI, lang="en")
    )

    assert reply.startswith("🔴 उच्च जोखिम · संभावित धोखाधड़ी")
    assert reply.endswith(locales.get("hi").disclaimer)


async def test_without_an_llm_the_scam_is_still_found(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> None:
    deps = pipeline_deps(llm=DisabledLLMClient())

    reply = await check(redis, adapter, locales, deps, fake_update(5, ELECTRICITY_EN))

    assert reply.startswith("🔴 High risk · Likely scam")


async def test_checker_server_down_gives_could_not_verify(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> None:
    deps = pipeline_deps(caller=FailingToolCaller())

    reply = await check(redis, adapter, locales, deps, fake_update(6, ELECTRICITY_EN))

    en = locales.get("en").verdict
    assert reply.startswith("⚪ Unknown · Could not verify")
    assert en.unavailable.replace("{checks}", en.checkers.analyze_scam_patterns) in reply


async def test_job_processed_twice_gives_one_verdict(
    adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory, store: InMemoryStore
) -> None:
    job = adapter.parse_update(fake_update(7, ELECTRICITY_EN), trace_id="t")
    assert isinstance(job, Job)
    deps = pipeline_deps()

    await process_job({DEPS_KEY: deps}, job)
    await process_job({DEPS_KEY: deps}, job)

    assert [action for action, _, _ in adapter.log] == ["verdict"]
    assert len(store.requests) == 1


async def test_logs_never_contain_the_message(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    stream = io.StringIO()
    configure_logging("DEBUG", stream)
    secret = ELECTRICITY_EN.replace("9876543210", "9123456789")

    await check(redis, adapter, locales, pipeline_deps(), fake_update(8, secret, user_id=424242))

    logs = caplog.text + stream.getvalue()
    assert "job_replied" in logs
    for fragment in ("electricity power will be disconnected", "9123456789", "424242"):
        assert fragment not in logs
