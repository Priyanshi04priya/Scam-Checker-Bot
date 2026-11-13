"""Audio and video checks end to end, with fake detectors and engines."""

import dataclasses
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from arq.connections import ArqRedis

from scambot.agent.executor import McpToolCaller
from scambot.core.config import AppConfig
from scambot.core.locales import Locales
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from scambot.mcp_server.checkers.provenance import ProvenanceChecker
from scambot.mcp_server.checkers.voice_clone import VoiceCloneChecker
from scambot.mcp_server.clients.sightengine import SightengineError, SynthScores
from scambot.mcp_server.clients.vision import WebPage
from scambot.mcp_server.server import CheckerDeps, build_server
from scambot.preprocess.audio import FfmpegAudio
from scambot.preprocess.keyframes import create_keyframes
from scambot.preprocess.media import MediaEngines
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    FakeAdapter,
    FakeAudio,
    FakeImageAnalyzer,
    FakeKeyframes,
    FakeLLMClient,
    FakeStt,
    FakeVoiceDetector,
    FakeWebSearch,
    InMemoryStore,
    KeywordEmbedder,
    fake_media_update,
    judge,
    seed,
    speech_vad,
)
from tests.integration.test_image_checks import analysis_of, checks_of, has_no_genuine_claim
from tests.integration.test_scam_text import check
from tests.media import MP4_BYTES, OGG_BYTES, is_empty
from tests.samples import (
    LAPTOP_CLAIM_HI,
    LAPTOP_FACTCHECK,
    LAPTOP_FACTCHECK_URL,
    LAPTOP_FORWARD_HI,
    UNRELATED_FACTCHECKS,
    VOICE_HARMLESS_TEXT,
    VOICE_SCAM_TEXT,
)


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


def with_engines(engines: MediaEngines, **changes: Any) -> MediaEngines:
    return dataclasses.replace(engines, **changes)


def voice_note(adapter: FakeAdapter, update_id: int = 1, lang: str = "en") -> dict[str, Any]:
    adapter.files["vo"] = OGG_BYTES
    update = fake_media_update(update_id, "voice", "vo", source="voice")
    update["lang"] = lang
    return update


def video(adapter: FakeAdapter, update_id: int = 1) -> dict[str, Any]:
    adapter.files["vid"] = MP4_BYTES
    return fake_media_update(update_id, "video", "vid", source="video", mime="video/mp4")


def deps_with_voice(checker_deps: CheckerDeps, detector: FakeVoiceDetector | None) -> CheckerDeps:
    old = checker_deps.voice
    voice = VoiceCloneChecker(detector, old._media, old._cfg, speech_vad())
    return dataclasses.replace(checker_deps, voice=voice)


@pytest.mark.parametrize("lang", ["en", "hi"])
async def test_cloned_voice_asking_for_money_is_a_likely_scam_with_a_note(
    lang: str,
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    voice_detector: FakeVoiceDetector,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    voice_detector.score_value = 0.78
    fake_llm.responses["analyze"] = analysis_of([], language=lang)
    engines = with_engines(media_engines, stt=FakeStt(VOICE_SCAM_TEXT, "en"))

    reply = await check(
        redis, adapter, locales, pipeline_deps(engines=engines), voice_note(adapter, lang=lang)
    )

    texts = locales.get(lang).verdict
    assert reply.startswith("🔴")
    assert f"{texts.bands.high} · {texts.labels.likely_scam}" in reply.splitlines()[0]
    assert texts.voice_note in reply
    assert has_no_genuine_claim(reply)
    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.label == "likely_scam"
    assert ("detect_voice_clone", "ok", "likely_synthetic") in [
        tuple(item) for item in row.verdict.contributing
    ]
    assert checks_of(store)["analyze_scam_patterns"].confidence >= 0.70
    assert is_empty(media_root)


async def test_a_low_voice_score_leaves_the_verdict_and_adds_no_note(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    voice_detector: FakeVoiceDetector,
    store: InMemoryStore,
) -> None:
    voice_detector.score_value = 0.2
    fake_llm.responses["analyze"] = analysis_of([])
    engines = with_engines(media_engines, stt=FakeStt(VOICE_SCAM_TEXT, "en"))

    reply = await check(
        redis, adapter, locales, pipeline_deps(engines=engines), voice_note(adapter)
    )

    assert reply.startswith("🔴")
    assert locales.get("en").verdict.voice_note not in reply
    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.label == "likely_scam"


async def test_without_a_detector_the_scam_check_still_decides(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    checker_deps: CheckerDeps,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    engines = with_engines(media_engines, stt=FakeStt(VOICE_SCAM_TEXT, "en"))
    caller = McpToolCaller(build_server(deps_with_voice(checker_deps, None)), 5)

    reply = await check(
        redis, adapter, locales, pipeline_deps(engines=engines, caller=caller), voice_note(adapter)
    )

    en = locales.get("en").verdict
    assert reply.startswith("🔴")
    assert en.unavailable.replace("{checks}", en.checkers.detect_voice_clone) in reply
    assert en.voice_note not in reply


async def test_a_very_short_clip_gives_no_note_and_no_failure(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    voice_detector: FakeVoiceDetector,
    store: InMemoryStore,
) -> None:
    voice_detector.score_value = 0.9
    fake_llm.responses["analyze"] = analysis_of([])
    engines = with_engines(
        media_engines, stt=FakeStt(VOICE_SCAM_TEXT, "en"), audio=FakeAudio(seconds=1.5)
    )

    reply = await check(
        redis, adapter, locales, pipeline_deps(engines=engines), voice_note(adapter)
    )

    assert locales.get("en").verdict.voice_note not in reply
    assert checks_of(store)["detect_voice_clone"].status == "inconclusive"
    assert voice_detector.calls == []


async def test_a_synthetic_voice_alone_is_possibly_ai(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    voice_detector: FakeVoiceDetector,
    store: InMemoryStore,
) -> None:
    voice_detector.score_value = 0.80
    fake_llm.responses["analyze"] = analysis_of([])
    engines = with_engines(media_engines, stt=FakeStt(VOICE_HARMLESS_TEXT, "en"))

    reply = await check(
        redis, adapter, locales, pipeline_deps(engines=engines), voice_note(adapter)
    )

    assert "Medium risk · Possibly AI-generated" in reply.splitlines()[0]
    assert locales.get("en").verdict.voice_note in reply
    assert has_no_genuine_claim(reply)
    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.label == "possibly_ai"


async def test_video_with_a_synthetic_frame_and_a_debunked_claim(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    analyzer: FakeImageAnalyzer,
    web_search: FakeWebSearch,
    factcheck_index: InMemoryFactcheckIndex,
    embedder: KeywordEmbedder,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    await seed(factcheck_index, embedder, LAPTOP_FACTCHECK, *UNRELATED_FACTCHECKS)
    fake_llm.responses["analyze"] = analysis_of([LAPTOP_CLAIM_HI], language="hi")
    fake_llm.responses["factcheck_relevance"] = judge()
    analyzer.by_name["kf_01.jpg"] = SynthScores(0.9, None)
    analyzer.by_name["kf_02.jpg"] = SightengineError("status_500")  # one frame failing is tolerated
    engines = with_engines(
        media_engines, stt=FakeStt(LAPTOP_FORWARD_HI, "hi"), keyframes=FakeKeyframes(3, size=320)
    )

    reply = await check(redis, adapter, locales, pipeline_deps(engines=engines), video(adapter))

    assert LAPTOP_FACTCHECK_URL in reply
    results = checks_of(store)
    assert results["detect_deepfake"].signal == "likely_synthetic"
    assert results["detect_deepfake"].details["top_frame"] == "kf_01.jpg"
    assert len(web_search.calls) == 3  # three key frames searched
    assert results["analyze_scam_patterns"].status == "ok"
    assert results["detect_voice_clone"].status == "ok"
    assert is_empty(media_root)


async def test_video_with_a_synthetic_frame_is_possibly_ai(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    analyzer: FakeImageAnalyzer,
    store: InMemoryStore,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    analyzer.by_name["kf_01.jpg"] = SynthScores(0.9, None)
    engines = with_engines(media_engines, keyframes=FakeKeyframes(3, size=320))

    reply = await check(redis, adapter, locales, pipeline_deps(engines=engines), video(adapter))

    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.label == "possibly_ai"
    assert "Medium risk" in reply.splitlines()[0]


async def test_silent_video_skips_the_voice_check_quietly(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    store: InMemoryStore,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    engines = with_engines(
        media_engines, audio=FakeAudio(error="no_audio"), keyframes=FakeKeyframes(2, size=320)
    )

    reply = await check(redis, adapter, locales, pipeline_deps(engines=engines), video(adapter))

    assert "detect_voice_clone" not in checks_of(store)
    assert locales.get("en").verdict.checkers.detect_voice_clone not in reply


class DelayedSearch:
    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def pages_with_matches(self, image: bytes, *, max_results: int) -> list[WebPage]:
        import asyncio  # noqa: PLC0415

        await asyncio.sleep(self._delay_s)
        return []


async def test_video_checks_run_in_parallel(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    checker_deps: CheckerDeps,
    analyzer: FakeImageAnalyzer,
    voice_detector: FakeVoiceDetector,
) -> None:
    delay = 0.3
    analyzer.delay_s = delay
    voice_detector.delay_s = delay
    old = checker_deps.provenance
    provenance = ProvenanceChecker(
        DelayedSearch(delay), old._fetcher, old._media, old._limits, old._sources
    )
    deps = dataclasses.replace(checker_deps, provenance=provenance)
    fake_llm.responses["analyze"] = analysis_of([])
    engines = with_engines(media_engines, keyframes=FakeKeyframes(3, size=320))
    caller = McpToolCaller(build_server(deps), 5)

    await check(
        redis, adapter, locales, pipeline_deps(engines=engines, caller=caller), video(adapter)
    )

    assert len(analyzer.spans) == 3
    first_end = min(end for _, end in analyzer.spans)
    assert max(start for start, _ in analyzer.spans) < first_end  # frames overlapped
    voice_start, _ = voice_detector.spans[0]
    assert voice_start < first_end  # the voice check ran alongside the frames
    window = max(end for _, end in analyzer.spans + voice_detector.spans) - min(
        start for start, _ in analyzer.spans + voice_detector.spans
    )
    assert window < delay * 4  # not one after the other (3 frames + voice = 4 delays)


async def test_a_tight_deadline_gives_a_partial_reply(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    media_engines: MediaEngines,
    fake_llm: FakeLLMClient,
    app_config: AppConfig,
    voice_detector: FakeVoiceDetector,
) -> None:
    voice_detector.delay_s = 1.0
    fake_llm.responses["analyze"] = analysis_of([])
    pipeline = app_config.thresholds.pipeline.model_copy(update={"deadline_s": 0.5})
    thresholds = app_config.thresholds.model_copy(update={"pipeline": pipeline})
    config = dataclasses.replace(app_config, thresholds=thresholds)
    engines = with_engines(media_engines, stt=FakeStt(VOICE_SCAM_TEXT, "en"))

    reply = await check(
        redis, adapter, locales, pipeline_deps(engines=engines, config=config), voice_note(adapter)
    )

    assert reply
    assert has_no_genuine_claim(reply)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_sixty_second_video_is_prepared_quickly(tmp_path: Path, app_config: AppConfig) -> None:
    import subprocess  # noqa: PLC0415

    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "original"
    subprocess.run(  # noqa: S603
        [
            ffmpeg, "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=60",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=60",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-f", "mp4", str(source),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    media = app_config.thresholds.media

    started = time.perf_counter()
    frames = create_keyframes(media.keyframes).extract(source, media.keyframes.max_keyframes)
    wav = FfmpegAudio(media.stt, ffmpeg).extract(source)
    elapsed = time.perf_counter() - started

    assert 0 < len(frames) <= 6
    assert wav.stat().st_size == pytest.approx(60 * 16000 * 2, rel=0.05)
    assert elapsed < 20
