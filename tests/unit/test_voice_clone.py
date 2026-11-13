"""Tests for the synthetic-voice checker."""

from pathlib import Path

import numpy as np
import pytest

from scambot.core.config import AppConfig, VoiceThresholds
from scambot.core.media_store import AUDIO_FILE, MediaStore
from scambot.mcp_server.checkers.voice_clone import VoiceCloneChecker, select_segment
from scambot.mcp_server.clients.voice_spoof import VoiceSpoofError
from scambot.mcp_server.media import MediaFiles
from scambot.preprocess.stt import SAMPLE_RATE
from tests.fakes import FakeVoiceDetector, speech_vad, write_tone

KEY = "job1"


def put_wav(store_root: Path, seconds: float) -> None:
    folder = store_root / KEY
    folder.mkdir(parents=True, exist_ok=True)
    write_tone(folder / AUDIO_FILE, seconds)


def build(
    config: AppConfig,
    store: MediaStore,
    detector: FakeVoiceDetector | None,
    *,
    spans: list[tuple[int, int]] | None = None,
    cfg: VoiceThresholds | None = None,
) -> VoiceCloneChecker:
    return VoiceCloneChecker(
        detector, MediaFiles(store), cfg or config.thresholds.voice, speech_vad(spans)
    )


@pytest.fixture
def cfg(app_config: AppConfig) -> VoiceThresholds:
    return app_config.thresholds.voice


async def test_high_score_is_likely_synthetic(
    app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    put_wav(media_root, 6)
    detector = FakeVoiceDetector(0.78)

    result = await build(app_config, media_store, detector).check(KEY, AUDIO_FILE)

    assert (result.status, result.signal, result.confidence) == ("ok", "likely_synthetic", 0.78)
    assert "78%" in result.evidence[0].summary
    assert result.details["detector"] == "fake:voice"
    assert result.details["analysed_s"] == 6.0


async def test_score_below_the_cut_off_is_no_signal_and_no_claim_of_authenticity(
    app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    put_wav(media_root, 6)

    result = await build(app_config, media_store, FakeVoiceDetector(0.55)).check(KEY, AUDIO_FILE)

    assert (result.status, result.signal) == ("ok", "none")
    summary = result.evidence[0].summary.lower()
    assert "real" not in summary
    assert "genuine" not in summary.replace("does not show the voice is genuine", "")


async def test_little_speech_is_inconclusive(
    app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    put_wav(media_root, 6)
    detector = FakeVoiceDetector()

    short = await build(app_config, media_store, detector, spans=[(0, 2 * SAMPLE_RATE)]).check(
        KEY, AUDIO_FILE
    )
    none = await build(app_config, media_store, detector, spans=[]).check(KEY, AUDIO_FILE)

    assert (short.status, short.details["reason"], short.details["speech_s"]) == (
        "inconclusive",
        "too_little_speech",
        2.0,
    )
    assert (none.status, none.details["reason"]) == ("inconclusive", "no_speech")
    assert detector.calls == []


async def test_missing_or_broken_audio(
    app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    sut = build(app_config, media_store, FakeVoiceDetector())
    missing = await sut.check(KEY, AUDIO_FILE)
    folder = media_root / KEY
    folder.mkdir(parents=True)
    (folder / AUDIO_FILE).write_bytes(b"not a wav")
    broken = await sut.check(KEY, AUDIO_FILE)

    assert (missing.status, missing.details["reason"]) == ("inconclusive", "no_audio")
    assert (broken.status, broken.error) == ("error", "unreadable_audio")


async def test_oversized_audio_is_an_error(
    app_config: AppConfig, media_store: MediaStore, media_root: Path, cfg: VoiceThresholds
) -> None:
    put_wav(media_root, 6)
    small = cfg.model_copy(update={"max_wav_bytes": 1000})

    result = await build(app_config, media_store, FakeVoiceDetector(), cfg=small).check(
        KEY, AUDIO_FILE
    )

    assert result.error == "too_large"


def test_the_densest_window_is_chosen(cfg: VoiceThresholds) -> None:
    total = 40 * SAMPLE_RATE
    audio = np.zeros(total, dtype=np.float32)
    spans = [(0, 2 * SAMPLE_RATE), (30 * SAMPLE_RATE, 38 * SAMPLE_RATE)]

    segment, speech_s = select_segment(audio, spans, cfg)

    assert len(segment) == cfg.window_s * SAMPLE_RATE
    assert speech_s == 8.0


async def test_a_long_clip_is_cut_to_the_window(
    app_config: AppConfig, media_store: MediaStore, media_root: Path, cfg: VoiceThresholds
) -> None:
    put_wav(media_root, 25)

    result = await build(app_config, media_store, FakeVoiceDetector(0.9)).check(KEY, AUDIO_FILE)

    assert result.details["analysed_s"] == cfg.window_s


async def test_no_detector_is_not_configured(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    result = await build(app_config, media_store, None).check(KEY, AUDIO_FILE)

    assert (result.status, result.error) == ("error", "not_configured")


async def test_detector_failure_is_an_error(
    app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    put_wav(media_root, 6)
    detector = FakeVoiceDetector(VoiceSpoofError("inference_failed"))

    result = await build(app_config, media_store, detector).check(KEY, AUDIO_FILE)

    assert (result.status, result.error) == ("error", "inference_failed")


async def test_slow_detector_times_out(
    app_config: AppConfig, media_store: MediaStore, media_root: Path, cfg: VoiceThresholds
) -> None:
    put_wav(media_root, 6)
    fast = cfg.model_copy(update={"timeout_s": 0.2})

    result = await build(
        app_config, media_store, FakeVoiceDetector(0.9, delay_s=0.6), cfg=fast
    ).check(KEY, AUDIO_FILE)

    assert (result.status, result.error) == ("error", "timeout")


@pytest.mark.parametrize(("raw", "expected"), [(1.7, 1.0), (-0.3, 0.0)])
async def test_scores_are_clamped(
    raw: float, expected: float, app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    put_wav(media_root, 6)

    result = await build(app_config, media_store, FakeVoiceDetector(raw)).check(KEY, AUDIO_FILE)

    assert result.details["score"] == expected


async def test_missing_vad_package_is_not_configured(
    app_config: AppConfig, media_store: MediaStore, media_root: Path
) -> None:
    put_wav(media_root, 6)

    def no_vad(_audio: object) -> list[tuple[int, int]]:
        raise ImportError

    sut = VoiceCloneChecker(
        FakeVoiceDetector(),
        MediaFiles(media_store),
        app_config.thresholds.voice,
        no_vad,
    )

    assert (await sut.check(KEY, AUDIO_FILE)).error == "not_configured"
