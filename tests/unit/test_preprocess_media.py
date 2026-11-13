"""Tests for the media orchestration, with fake engines."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from scambot.core.contracts import MediaFeatures
from scambot.core.media import MediaRejected
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.hashing import file_sha256
from scambot.preprocess.media import MediaEngines, compose_text, detect_language, extract_media
from tests.fakes import BrokenEngine, FakeKeyframes, FakeOcr, FakeStt, SlowEngine
from tests.media import MP4_BYTES, OGG_BYTES, png_bytes
from tests.samples import ELECTRICITY_HI


def with_engines(engines: MediaEngines, **changes: Any) -> MediaEngines:
    return dataclasses.replace(engines, **changes)


def write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "original"
    path.write_bytes(data)
    return path


async def test_image_gives_text_and_hashes(tmp_path: Path, media_engines: MediaEngines) -> None:
    path = write(tmp_path, png_bytes())

    features = await extract_media(
        "image", path, with_engines(media_engines, ocr=FakeOcr("Your KYC expires"))
    )

    assert features.mime_type == "image/png"
    assert features.sha256 == file_sha256(path)
    assert features.ocr_text == "Your KYC expires"
    assert features.phash is not None
    assert features.errors == []


async def test_voice_gives_transcript_and_language(
    tmp_path: Path, media_engines: MediaEngines
) -> None:
    path = write(tmp_path, OGG_BYTES)
    engines = with_engines(media_engines, stt=FakeStt(ELECTRICITY_HI, "hi"))

    features = await extract_media("voice", path, engines)

    assert features.transcript == ELECTRICITY_HI
    assert features.speech_language == "hi"
    assert features.phash is None


async def test_video_combines_keyframes_ocr_and_speech(
    tmp_path: Path, media_engines: MediaEngines
) -> None:
    path = write(tmp_path, MP4_BYTES)
    ocr = FakeOcr("Free laptop\nfor all")
    engines = with_engines(
        media_engines, ocr=ocr, stt=FakeStt("spoken words", "hi"), keyframes=FakeKeyframes(5)
    )

    features = await extract_media("video", path, engines)

    assert len(features.keyframe_phashes) == 5
    assert features.phash == features.keyframe_phashes[0]
    assert len(ocr.calls) == 3  # only the first max_video_frames frames are read
    assert features.ocr_text.count("Free laptop") == 1  # repeated lines are kept once
    assert features.transcript == "spoken words"


async def test_a_failing_engine_is_recorded_and_the_rest_kept(
    tmp_path: Path, media_engines: MediaEngines
) -> None:
    path = write(tmp_path, MP4_BYTES)
    engines = with_engines(
        media_engines,
        ocr=BrokenEngine(RuntimeError("boom")),
        stt=BrokenEngine(MediaToolError("no_audio")),
        keyframes=FakeKeyframes(2),
    )

    features = await extract_media("video", path, engines)

    assert sorted(features.errors) == ["ocr:failed", "ocr:failed", "stt:no_audio"]
    assert len(features.keyframe_phashes) == 2


async def test_unavailable_engines_are_noted(tmp_path: Path, media_engines: MediaEngines) -> None:
    path = write(tmp_path, OGG_BYTES)
    engines = with_engines(media_engines, stt=BrokenEngine(MediaToolError("unavailable")))

    features = await extract_media("audio", path, engines)

    assert features.errors == ["stt:unavailable"]


async def test_slow_engine_times_out_alone(tmp_path: Path, media_engines: MediaEngines) -> None:
    path = write(tmp_path, png_bytes())
    limits = media_engines.limits.model_copy(
        update={"ocr": media_engines.limits.ocr.model_copy(update={"timeout_s": 0.05})}
    )
    engines = with_engines(media_engines, ocr=SlowEngine(0.5), limits=limits)

    features = await extract_media("image", path, engines)

    assert features.errors == ["ocr:timeout"]
    assert features.phash is not None


async def test_overall_budget_keeps_partial_results(
    tmp_path: Path, media_engines: MediaEngines
) -> None:
    path = write(tmp_path, MP4_BYTES)
    limits = media_engines.limits.model_copy(update={"preprocess_timeout_s": 0.2})
    engines = with_engines(
        media_engines,
        stt=SlowEngine(1.0),
        keyframes=FakeKeyframes(2),
        ocr=FakeOcr("seen text"),
        limits=limits,
    )

    features = await extract_media("video", path, engines)

    assert "preprocess:timeout" in features.errors
    assert features.ocr_text == "seen text"
    assert features.transcript == ""


async def test_invalid_file_is_rejected(tmp_path: Path, media_engines: MediaEngines) -> None:
    path = write(tmp_path, b"plain text")

    with pytest.raises(MediaRejected) as raised:
        await extract_media("image", path, media_engines)
    assert raised.value.reason == "unsupported_type"


def test_compose_text_drops_empty_parts() -> None:
    features = MediaFeatures(
        mime_type="image/png", size_bytes=1, sha256="0" * 64, ocr_text="from image", transcript=""
    )

    assert compose_text("caption", features) == "caption\n\nfrom image"
    assert compose_text("", features) == "from image"
    assert compose_text("only caption", None) == "only caption"
    assert compose_text("", None) == ""


async def test_detect_language(media_engines: MediaEngines) -> None:
    assert await detect_language(ELECTRICITY_HI, None, media_engines) == "hi"
    assert await detect_language("", "hi", media_engines) == "hi"
    assert await detect_language("12", "ta", media_engines) == "ta"

    class Failing:
        def detect(self, text: str) -> str:
            raise RuntimeError

    assert (
        await detect_language("hello", "en", with_engines(media_engines, language=Failing()))
        == "en"
    )
