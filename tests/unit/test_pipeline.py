"""Tests for the job pipeline and the worker settings."""

import asyncio
import dataclasses
import hashlib
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from aiogram import Bot

from scambot.agent.analyze import Analysis
from scambot.core.config import AppConfig
from scambot.core.contracts import Job, JobKind
from scambot.core.llm import DisabledLLMClient, PromptLibrary
from scambot.core.locales import Locales
from scambot.core.media_store import LocalMediaStore, MediaStore, media_key
from scambot.core.redis import PROCESS_JOB
from scambot.mcp_server.checkers.scam import ScamClassification
from scambot.preprocess.entities import HeadRedirectExpander, NoopExpander
from scambot.preprocess.hashing import text_sha256
from scambot.preprocess.media import MediaEngines
from scambot.worker import settings as worker_settings
from scambot.worker.cron import INGEST_KEY, MEDIA_SWEEP_KEY, SweepDeps, sweep_media
from scambot.worker.pipeline import DEPS_KEY, PipelineDeps, process_job
from scambot.worker.settings import WorkerSettings
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    FailingToolCaller,
    FakeAdapter,
    FakeLLMClient,
    FakeOcr,
    FakeStt,
    InMemoryStore,
    StoredRequest,
)
from tests.media import EXE_BYTES, OGG_BYTES, is_empty, png_bytes
from tests.samples import ELECTRICITY_EN, ELECTRICITY_HI, ORDINARY


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
    jobs = {job.name: job for job in WorkerSettings.cron_jobs}
    assert set(jobs) == {"ingest_factchecks", "sweep_media"}
    sweep = jobs["sweep_media"]
    assert sweep.minute == set(range(0, 60, 10))
    assert sweep.run_at_startup is True
    assert sweep.unique is True
    job = jobs["ingest_factchecks"]
    assert job.name == "ingest_factchecks"
    assert (job.hour, job.minute) == ({2}, {0})
    assert job.unique is True
    assert job.timeout_s == 1800
    assert job.max_tries == 1
    assert job.run_at_startup is False
    assert WorkerSettings.timezone.utcoffset(None) == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("expand", ["true", "false"])
async def test_worker_startup_and_shutdown(
    monkeypatch: pytest.MonkeyPatch, expand: str, media_root: Path
) -> None:
    monkeypatch.setenv("MEDIA_DIR", str(media_root))
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
    pool = context[worker_settings.MEDIA_POOL_KEY]
    assert deps.media.pool is pool
    assert context[MEDIA_SWEEP_KEY].older_than == timedelta(minutes=45)
    assert isinstance(deps.media_store, LocalMediaStore)
    await worker_settings.shutdown(context)
    with pytest.raises(RuntimeError):
        pool.submit(print)
    await worker_settings.shutdown({})


async def test_worker_startup_with_llm_key(
    monkeypatch: pytest.MonkeyPatch, media_root: Path
) -> None:
    monkeypatch.setenv("MEDIA_DIR", str(media_root))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:" + "A" * 35)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    context: dict[str, Any] = {}

    await worker_settings.startup(context)

    assert context[worker_settings.ANTHROPIC_KEY] is not None
    await worker_settings.shutdown(context)


class WarmableStt(FakeStt):
    warmed = False

    def warm_up(self) -> None:
        WarmableStt.warmed = True


async def test_sweep_media_removes_an_aged_directory(media_root: Path) -> None:
    store = LocalMediaStore(media_root)
    old = media_root / "old"
    old.mkdir()
    (old / "original").write_bytes(b"x")
    long_ago = 1_000.0
    os.utime(old / "original", (long_ago, long_ago))
    os.utime(old, (long_ago, long_ago))
    (media_root / "fresh").mkdir()

    removed = await sweep_media({MEDIA_SWEEP_KEY: SweepDeps(store, timedelta(minutes=45))})

    assert removed == 1
    assert os.listdir(media_root) == ["fresh"]


async def test_worker_warms_up_speech_to_text(
    monkeypatch: pytest.MonkeyPatch, media_root: Path, media_engines: MediaEngines
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:" + "A" * 35)
    monkeypatch.setenv("MEDIA_DIR", str(media_root))
    warm = dataclasses.replace(media_engines, stt=WarmableStt())
    monkeypatch.setattr(worker_settings, "create_engines", lambda *_a: warm)
    context: dict[str, Any] = {}

    await worker_settings.startup(context)
    media_engines.pool.shutdown(wait=True)

    assert WarmableStt.warmed
    await worker_settings.shutdown(context)


# --- media jobs ----------------------------------------------------------------------------


def media_job(
    ref: str = "f1", kind: JobKind = "image", caption: str | None = "caption", lang: str = "en"
) -> Job:
    return Job(
        job_id="telegram:9",
        trace_id="trace-9",
        channel="telegram",
        chat_id="7",
        user_hash="h",
        kind=kind,
        text=caption,
        media_ref=ref,
        preferred_lang=lang,
        received_at=datetime.now(UTC),
    )


def hindi_analysis_response() -> Analysis:
    return Analysis(
        claims=[],
        reply_language="hi",
        claimed_date=None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )


def row_of(store: InMemoryStore) -> StoredRequest:
    (row,) = store.requests.values()
    return row


async def test_image_is_read_checked_stored_and_deleted(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    store: InMemoryStore,
    fake_llm: FakeLLMClient,
    media_engines: MediaEngines,
    media_root: Path,
    locales: Locales,
) -> None:
    data = png_bytes()
    adapter.files["f1"] = data
    fake_llm.responses["analyze"] = hindi_analysis_response()
    engines = dataclasses.replace(media_engines, ocr=FakeOcr(ELECTRICITY_HI))

    await process_job(ctx(pipeline_deps(engines=engines)), media_job(caption="Forwarded"))

    assert only_reply(adapter).startswith(f"🔴 {locales.get('hi').verdict.bands.high}")
    row = row_of(store)
    assert row.status == "replied"
    assert row.verdict is not None
    assert row.verdict.label == "likely_scam"
    assert row.sha256 == hashlib.sha256(data).hexdigest()
    assert isinstance(row.phash, int)
    assert row.request.sha256 is None
    assert row.request.caption_hash == text_sha256("Forwarded")
    assert row.language == "hi"
    assert is_empty(media_root)


async def test_language_comes_from_the_text_without_analysis(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    store: InMemoryStore,
    media_engines: MediaEngines,
) -> None:
    adapter.files["f1"] = OGG_BYTES
    engines = dataclasses.replace(media_engines, stt=FakeStt(ELECTRICITY_HI, "hi"))

    await process_job(ctx(pipeline_deps(engines=engines)), media_job(kind="voice", caption=None))

    assert row_of(store).language == "hi"


async def test_executable_renamed_as_photo_gets_the_rejection_reply(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    store: InMemoryStore,
    locales: Locales,
    media_root: Path,
) -> None:
    adapter.files["f1"] = EXE_BYTES

    await process_job(ctx(pipeline_deps()), media_job())

    assert only_reply(adapter) == locales.get("en").rejected.unsupported_type
    row = row_of(store)
    assert row.status == "replied"
    assert row.checks == []
    assert row.verdict is None
    assert is_empty(media_root)


async def test_streamed_bytes_over_the_cap_are_rejected(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    locales: Locales,
    monkeypatch: pytest.MonkeyPatch,
    media_root: Path,
) -> None:
    monkeypatch.setattr("scambot.worker.pipeline.MAX_MEDIA_BYTES", 100)
    adapter.files["f1"] = b"x" * 101

    await process_job(ctx(pipeline_deps()), media_job())

    assert only_reply(adapter) == locales.get("en").rejected.too_large
    assert is_empty(media_root)


async def test_download_failing_twice_is_unreadable(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, locales: Locales
) -> None:
    adapter.files["f1"] = png_bytes()
    adapter.fail_download = 2

    await process_job(ctx(pipeline_deps()), media_job())

    assert adapter.downloads == 2
    assert only_reply(adapter) == locales.get("en").rejected.unreadable


async def test_download_failing_once_is_retried(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, store: InMemoryStore
) -> None:
    adapter.files["f1"] = png_bytes()
    adapter.fail_download = 1

    await process_job(ctx(pipeline_deps()), media_job())

    assert adapter.downloads == 2
    assert row_of(store).sha256 is not None


async def test_media_job_without_a_reference_is_unreadable(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, locales: Locales
) -> None:
    job = media_job().model_copy(update={"media_ref": None})

    await process_job(ctx(pipeline_deps()), job)

    assert only_reply(adapter) == locales.get("en").rejected.unreadable


async def test_slow_download_is_unreadable_not_a_job_timeout(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    locales: Locales,
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stall(*_args: Any, **_kwargs: Any) -> Path:
        await asyncio.sleep(5)
        raise AssertionError

    monkeypatch.setattr(adapter, "download_media", stall)
    media = app_config.thresholds.media.model_copy(update={"download_timeout_s": 0.05})
    thresholds = app_config.thresholds.model_copy(update={"media": media})
    config = dataclasses.replace(app_config, thresholds=thresholds)

    await process_job(ctx(pipeline_deps(config=config)), media_job())

    assert only_reply(adapter) == locales.get("en").rejected.unreadable


async def test_media_is_deleted_when_analysis_blows_up(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.files["f1"] = png_bytes()

    def explode(*_args: Any) -> Any:
        msg = "bug"
        raise RuntimeError(msg)

    monkeypatch.setattr("scambot.worker.pipeline.build_plan", explode)

    with pytest.raises(RuntimeError):
        await process_job(ctx(pipeline_deps()), media_job())

    assert is_empty(media_root)


async def test_media_is_deleted_when_the_deadline_passes(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    media_root: Path,
    app_config: AppConfig,
    prompts: PromptLibrary,
) -> None:
    adapter.files["f1"] = png_bytes()
    deps = pipeline_deps(llm=SlowExplanation(prompts), config=tiny_deadline(app_config, 0.5))

    await process_job(ctx(deps), media_job())

    assert adapter.messages
    assert is_empty(media_root)


async def test_media_is_deleted_for_an_already_replied_request(
    pipeline_deps: PipelineDepsFactory,
    adapter: FakeAdapter,
    media_store: MediaStore,
    media_root: Path,
) -> None:
    adapter.files["f1"] = png_bytes()
    deps = pipeline_deps()
    await process_job(ctx(deps), media_job())
    key = media_key("telegram:9")
    (media_root / key).mkdir()
    (media_root / key / "leftover").write_bytes(b"x")

    await process_job(ctx(deps), media_job())

    assert is_empty(media_root)


async def test_a_failing_final_delete_is_logged_not_raised(
    pipeline_deps: PipelineDepsFactory, adapter: FakeAdapter, tmp_path: Path
) -> None:
    adapter.files["f1"] = png_bytes()

    class FlakyStore(LocalMediaStore):
        deletes = 0

        def delete(self, key: str) -> None:
            self.deletes += 1
            if self.deletes > 1:  # the clean-slate delete works, the final one fails
                msg = "disk"
                raise OSError(msg)
            super().delete(key)

    deps = pipeline_deps(store_override=FlakyStore(tmp_path / "flaky"))

    await process_job(ctx(deps), media_job())

    assert adapter.messages


async def test_text_job_gets_a_detected_language_and_no_caption_hash(
    pipeline_deps: PipelineDepsFactory, store: InMemoryStore
) -> None:
    await process_job(ctx(pipeline_deps()), a_job(ELECTRICITY_HI, lang="en"))

    row = row_of(store)
    assert row.request.caption_hash is None
    assert row.request.sha256 is not None
    assert row.language == "hi"
