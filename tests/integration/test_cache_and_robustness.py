"""Reusing earlier verdicts and keeping the worker robust: deadline, retries and cancellation.

The retry behaviour of the real queue is exercised at the end with ``arq.Worker`` running over
the in-memory Redis.
"""

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from arq import Retry
from arq.connections import ArqRedis
from mcp.server.mcpserver import MCPServer
from pydantic import SecretStr
from sqlalchemy import select

from scambot.agent.analyze import Analysis
from scambot.agent.executor import McpToolCaller, ToolCaller, ToolSession
from scambot.core.cache import CheckerCache
from scambot.core.config import AppConfig
from scambot.core.contracts import Job
from scambot.core.db import create_engine, create_session_factory
from scambot.core.locales import Locales
from scambot.core.models import Base, Request
from scambot.core.redis import PROCESS_JOB
from scambot.core.runtime import RuntimeConfig, RuntimeDefaults
from scambot.core.store import SqlRequestStore
from scambot.mcp_server.clients.vision import VisionError
from scambot.preprocess.media import MediaEngines
from scambot.worker.pipeline import DEPS_KEY, PipelineDeps, process_job
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    CancellingCaller,
    CountingCaller,
    FakeAdapter,
    FakeImageAnalyzer,
    FakeLLMClient,
    FakeWebSearch,
    FlakyAdapter,
    InMemorySettingsStore,
    InMemoryStore,
    fake_media_update,
    mismatch_reply,
)
from tests.integration.test_scam_text import check, run_worker
from tests.media import is_empty, recompressed, textured_jpeg

CAPTION = "Delhi flood today"
TTLS = dict.fromkeys(
    (
        "analyze_scam_patterns",
        "fact_check_claim",
        "check_provenance",
        "check_caption_context",
        "detect_deepfake",
        "detect_voice_clone",
    ),
    3600,
)


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


def analysis(_variables: Mapping[str, str]) -> Analysis:
    return Analysis(
        claims=[],
        reply_language="en",
        claimed_date=None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )


@pytest.fixture(autouse=True)
def _llm(fake_llm: FakeLLMClient) -> None:
    fake_llm.responses["analyze"] = analysis
    fake_llm.responses["caption_compare"] = mismatch_reply()


@pytest.fixture
async def sql_store(tmp_path: Path) -> AsyncIterator[SqlRequestStore]:
    engine = create_engine(SecretStr(f"sqlite+aiosqlite:///{tmp_path}/t.db"))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield SqlRequestStore(create_session_factory(engine))
    await engine.dispose()


@pytest.fixture
def cache(redis: ArqRedis) -> CheckerCache:
    return CheckerCache(redis, TTLS, 3600)  # type: ignore[arg-type]


def picture(
    adapter: FakeAdapter,
    update_id: int,
    data: bytes,
    *,
    caption: str | None = CAPTION,
    lang: str | None = "en",
) -> dict[str, Any]:
    adapter.files[f"img{update_id}"] = data
    return fake_media_update(update_id, "image", f"img{update_id}", caption=caption, lang=lang)


def job_of(adapter: FakeAdapter, update: dict[str, Any]) -> Job:
    job = adapter.parse_update(update, trace_id="trace")
    assert isinstance(job, Job)
    return job


def ctx(deps: PipelineDeps, job_try: int = 1) -> dict[str, Any]:
    return {DEPS_KEY: deps, "job_try": job_try}


def with_deadline(config: AppConfig, deadline_s: float) -> AppConfig:
    pipeline = config.thresholds.pipeline.model_copy(update={"deadline_s": deadline_s})
    return dataclasses.replace(
        config, thresholds=config.thresholds.model_copy(update={"pipeline": pipeline})
    )


async def sql_rows(store: SqlRequestStore) -> list[Request]:
    async with store._sessions() as session:
        return list((await session.scalars(select(Request).order_by(Request.created_at))).all())


# --- the verdict cache ---------------------------------------------------------------------


async def test_a_recompressed_forward_is_answered_from_the_earlier_verdict(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    mcp_server: MCPServer,
    fake_llm: FakeLLMClient,
    media_engines: MediaEngines,
    sql_store: SqlRequestStore,
    cache: CheckerCache,
    web_search: FakeWebSearch,
    analyzer: FakeImageAnalyzer,
    media_root: Path,
) -> None:
    original = textured_jpeg()
    counting = CountingCaller(McpToolCaller(mcp_server, 5))
    deps = pipeline_deps(request_store=sql_store, caller=counting, cache=cache)
    first = await check(redis, adapter, locales, deps, picture(adapter, 1, original))
    llm_calls, tools = len(fake_llm.calls), len(counting.tools)
    ocr_calls = len(getattr(media_engines.ocr, "calls", []))
    searches, scans = len(web_search.calls), len(analyzer.calls)
    assert tools > 0

    started = time.perf_counter()
    second = await check(redis, adapter, locales, deps, picture(adapter, 2, recompressed(original)))
    elapsed = time.perf_counter() - started

    assert locales.get("en").cached.split("{")[0] in second
    assert first.splitlines()[0] in second
    assert len(fake_llm.calls) == llm_calls
    assert len(counting.tools) == tools
    assert len(getattr(media_engines.ocr, "calls", [])) == ocr_calls
    assert (len(web_search.calls), len(analyzer.calls)) == (searches, scans)
    first_row, second_row = await sql_rows(sql_store)
    assert first_row.cache_of is None
    assert (second_row.cache_of, second_row.status, second_row.cost_usd) == (
        first_row.id,
        "replied",
        Decimal(0),
    )
    assert elapsed < 3.0
    assert is_empty(media_root)


async def test_the_same_picture_with_another_caption_reuses_only_the_picture_checks(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    store: InMemoryStore,
    cache: CheckerCache,
    web_search: FakeWebSearch,
    analyzer: FakeImageAnalyzer,
) -> None:
    data = textured_jpeg()
    deps = pipeline_deps(cache=cache)
    await check(redis, adapter, locales, deps, picture(adapter, 1, data))
    llm_calls = len(fake_llm.calls)

    reply = await check(
        redis, adapter, locales, deps, picture(adapter, 2, data, caption="Fire in Mumbai")
    )

    assert locales.get("en").cached.split("{")[0] not in reply
    assert (len(web_search.calls), len(analyzer.calls)) == (1, 1)  # served from the checker cache
    assert len(fake_llm.calls) > llm_calls  # the caption was compared again
    second = store.requests["telegram:2"]
    assert second.cache_of is None
    assert {c.result.checker: c.cached for c in second.checks}["detect_deepfake"] is True


async def test_a_hindi_speaker_gets_the_cached_verdict_in_hindi(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    data = textured_jpeg()
    deps = pipeline_deps()
    await check(redis, adapter, locales, deps, picture(adapter, 1, data))

    reply = await check(redis, adapter, locales, deps, picture(adapter, 2, data, lang="hi"))

    hindi = locales.get("hi")
    label = store.requests["telegram:1"].verdict
    assert label is not None
    assert hindi.cached.split("{")[0] in reply
    assert getattr(hindi.verdict.labels, label.label) in reply
    assert store.requests["telegram:2"].cache_of == store.requests["telegram:1"].id


async def test_without_a_preferred_language_the_original_reply_language_is_kept(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
) -> None:
    data = textured_jpeg()
    deps = pipeline_deps()
    first = await check(
        redis, adapter, locales, deps, picture(adapter, 1, data, caption=None, lang="hi")
    )

    reply = await check(
        redis, adapter, locales, deps, picture(adapter, 2, data, caption=None, lang=None)
    )

    assert locales.get("hi").cached.split("{")[0] in reply
    assert "Result from" not in reply
    assert first.split("\n", 1)[0] in reply


async def test_a_verdict_made_during_an_outage_is_not_replayed(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
    cache: CheckerCache,
    web_search: FakeWebSearch,
) -> None:
    data = textured_jpeg()
    deps = pipeline_deps(cache=cache)
    web_search.pages = VisionError("status_500")
    await check(redis, adapter, locales, deps, picture(adapter, 1, data))
    web_search.pages = []

    reply = await check(redis, adapter, locales, deps, picture(adapter, 2, data))

    assert locales.get("en").cached.split("{")[0] not in reply
    assert len(web_search.calls) == 2  # asked again: the failure was not cached either
    assert store.requests["telegram:2"].cache_of is None


async def test_a_short_window_in_the_settings_table_turns_reuse_off(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
    web_search: FakeWebSearch,
) -> None:
    settings = InMemorySettingsStore({"cache_window_days": 1e-9})
    runtime = RuntimeConfig(settings, redis, RuntimeDefaults(20, 7), 60)
    data = textured_jpeg()
    deps = pipeline_deps(runtime=runtime)
    await check(redis, adapter, locales, deps, picture(adapter, 1, data))
    await asyncio.sleep(0.01)

    await check(redis, adapter, locales, deps, picture(adapter, 2, data))

    assert len(web_search.calls) == 2
    assert store.requests["telegram:2"].cache_of is None


async def test_a_store_failure_during_lookup_is_only_a_miss(
    adapter: FakeAdapter, pipeline_deps: PipelineDepsFactory, store: InMemoryStore
) -> None:
    store.fail = True

    await process_job(ctx(pipeline_deps()), job_of(adapter, picture(adapter, 1, textured_jpeg())))

    assert len(adapter.messages) == 1


# --- the deadline --------------------------------------------------------------------------


async def test_a_slow_check_gives_a_partial_reply_that_is_never_reused(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    app_config: AppConfig,
    analyzer: FakeImageAnalyzer,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    data = textured_jpeg()
    analyzer.delay_s = 5
    slow = pipeline_deps(config=with_deadline(app_config, 2.0))

    reply = await check(redis, adapter, locales, slow, picture(adapter, 1, data))

    en = locales.get("en").verdict
    assert en.unavailable.split("{")[0] in reply
    assert en.checkers.detect_deepfake in reply
    first = store.requests["telegram:1"]
    assert (first.status, first.verdict is not None) == ("timeout", True)
    assert is_empty(media_root)

    analyzer.delay_s = 0
    again = await check(redis, adapter, locales, pipeline_deps(), picture(adapter, 2, data))

    assert locales.get("en").cached.split("{")[0] not in again
    assert store.requests["telegram:2"].cache_of is None


async def test_nothing_finished_before_the_deadline_gets_the_apology(
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    app_config: AppConfig,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    deps = pipeline_deps(config=with_deadline(app_config, 0.001))

    await process_job(ctx(deps), job_of(adapter, picture(adapter, 1, textured_jpeg())))

    assert next(iter(adapter.messages.values())) == locales.get("en").timeout
    assert status_of(store) == "timeout"
    assert is_empty(media_root)


# --- a worker that is killed or fails ------------------------------------------------------


async def test_a_cancelled_job_leaves_nothing_behind_and_runs_again_once(
    adapter: FakeAdapter,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    job = job_of(adapter, picture(adapter, 1, textured_jpeg()))

    with pytest.raises(asyncio.CancelledError):
        await process_job(ctx(pipeline_deps(caller=CancellingCaller()), 1), job)

    assert adapter.messages == {}
    assert status_of(store) == "received"
    assert is_empty(media_root)

    await process_job(ctx(pipeline_deps(), 2), job)
    assert len(adapter.messages) == 1
    assert status_of(store) == "replied"

    await process_job(ctx(pipeline_deps(), 3), job)
    assert len(adapter.messages) == 1


def status_of(store: InMemoryStore) -> str:
    return str(store.requests["telegram:1"].status)


def flaky(pipeline_deps: PipelineDepsFactory, adapter_: FlakyAdapter) -> PipelineDeps:
    return dataclasses.replace(pipeline_deps(), adapters={"telegram": adapter_})


async def test_a_failed_delivery_is_retried_then_succeeds(
    pipeline_deps: PipelineDepsFactory, store: InMemoryStore
) -> None:
    channel = FlakyAdapter(fail_sends=1)
    deps = flaky(pipeline_deps, channel)
    job = job_of(channel, picture(channel, 1, textured_jpeg()))

    with pytest.raises(Retry):
        await process_job(ctx(deps, 1), job)
    assert status_of(store) == "failed"

    await process_job(ctx(deps, 2), job)

    assert status_of(store) == "replied"
    assert len(channel.messages) == 1


async def test_the_last_failed_try_apologises_and_raises_the_original_error(
    pipeline_deps: PipelineDepsFactory, store: InMemoryStore
) -> None:
    channel = FlakyAdapter(fail_sends=10)
    deps = flaky(pipeline_deps, channel)
    job = job_of(channel, picture(channel, 1, textured_jpeg()))

    with pytest.raises(ConnectionError):
        await process_job(ctx(deps, 3), job)

    assert channel.send_attempts == 2  # the verdict, then the apology
    assert status_of(store) == "failed"


class CancelFirstSession(ToolCaller):
    """Cancels the first run, as a killed worker would, and works afterwards."""

    def __init__(self, inner: ToolCaller) -> None:
        self._inner = inner
        self.sessions = 0

    def session(self) -> AbstractAsyncContextManager[ToolSession]:
        self.sessions += 1
        return CancellingCaller().session() if self.sessions == 1 else self._inner.session()


async def test_the_queue_runs_a_cancelled_job_again(
    redis: ArqRedis,
    adapter: FakeAdapter,
    pipeline_deps: PipelineDepsFactory,
    mcp_server: MCPServer,
    store: InMemoryStore,
) -> None:
    caller = CancelFirstSession(McpToolCaller(mcp_server, 5))
    deps = pipeline_deps(caller=caller)
    job = job_of(adapter, picture(adapter, 1, textured_jpeg()))
    await redis.enqueue_job(PROCESS_JOB, job, _job_id=job.job_id)

    for _ in range(3):
        await run_worker(redis, deps)
        if status_of(store) == "replied":
            break

    assert status_of(store) == "replied"
    assert len(adapter.messages) == 1
