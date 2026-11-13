"""Media end to end: intake, queue, worker with fake extraction engines, real checkers."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from arq.connections import ArqRedis

from scambot.agent.analyze import Analysis
from scambot.api.intake import Intake, Outcome
from scambot.core.locales import Locales
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from scambot.preprocess.media import MediaEngines
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    FakeAdapter,
    FakeKeyframes,
    FakeLLMClient,
    FakeOcr,
    FakeStt,
    InMemoryStore,
    KeywordEmbedder,
    fake_media_update,
    judge,
    seed,
)
from tests.integration.test_scam_text import check
from tests.media import EXE_BYTES, MP4_BYTES, OGG_BYTES, is_empty, png_bytes
from tests.samples import (
    ELECTRICITY_HI,
    LAPTOP_CLAIM_HI,
    LAPTOP_FACTCHECK,
    LAPTOP_FACTCHECK_URL,
    LAPTOP_FORWARD_HI,
    UNRELATED_FACTCHECKS,
)


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


def hindi(claims: list[str]) -> Any:
    def answer(_variables: Mapping[str, str]) -> Analysis:
        return Analysis(
            claims=claims,
            reply_language="hi",
            claimed_date=None,
            claimed_place=None,
            extra_checkers=[],
            reason="",
        )

    return answer


def with_engines(engines: MediaEngines, **changes: Any) -> MediaEngines:
    return dataclasses.replace(engines, **changes)


async def test_hindi_scam_screenshot_is_high_risk(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    adapter.files["img"] = png_bytes()
    fake_llm.responses["analyze"] = hindi([])
    deps = pipeline_deps(engines=with_engines(media_engines, ocr=FakeOcr(ELECTRICITY_HI)))

    reply = await check(redis, adapter, locales, deps, fake_media_update(1, "image", "img"))

    assert reply.startswith(f"🔴 {locales.get('hi').verdict.bands.high}")
    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.label == "likely_scam"
    assert is_empty(media_root)


async def test_voice_note_scam_is_high_risk(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    media_root: Path,
) -> None:
    adapter.files["vo"] = OGG_BYTES
    deps = pipeline_deps(engines=with_engines(media_engines, stt=FakeStt(ELECTRICITY_HI, "hi")))
    update = fake_media_update(2, "voice", "vo", source="voice")

    reply = await check(redis, adapter, locales, deps, update)

    assert reply.startswith("🔴")
    assert is_empty(media_root)


async def test_video_with_a_debunked_claim_links_the_fact_check(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    factcheck_index: InMemoryFactcheckIndex,
    embedder: KeywordEmbedder,
    media_root: Path,
) -> None:
    await seed(factcheck_index, embedder, LAPTOP_FACTCHECK, *UNRELATED_FACTCHECKS)
    adapter.files["vid"] = MP4_BYTES
    fake_llm.responses["analyze"] = hindi([LAPTOP_CLAIM_HI])
    fake_llm.responses["factcheck_relevance"] = judge()
    engines = with_engines(
        media_engines, stt=FakeStt(LAPTOP_FORWARD_HI, "hi"), keyframes=FakeKeyframes(2)
    )
    update = fake_media_update(3, "video", "vid", source="video", mime="video/mp4")

    reply = await check(redis, adapter, locales, pipeline_deps(engines=engines), update)

    assert reply.startswith("🔴")
    assert LAPTOP_FACTCHECK_URL in reply
    assert is_empty(media_root)


async def test_declared_oversized_file_is_rejected_at_intake(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales
) -> None:
    update = fake_media_update(4, "video", "big", size=30 * 1024 * 1024, source="video")

    assert await Intake(adapter, redis, locales).handle(update) is Outcome.REJECTED

    assert adapter.log == [("text", "7", locales.get("en").rejected.too_large)]
    assert await redis.queued_jobs() == []


async def test_executable_renamed_as_a_photo_is_rejected_by_the_worker(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_root: Path,
) -> None:
    adapter.files["bad"] = EXE_BYTES

    reply = await check(
        redis, adapter, locales, pipeline_deps(), fake_media_update(5, "image", "bad")
    )

    assert reply == locales.get("en").rejected.unsupported_type
    assert is_empty(media_root)
