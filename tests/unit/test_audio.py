"""Tests for sound-track extraction."""

import shutil
from pathlib import Path

import pytest

from scambot.core.config import AppConfig
from scambot.preprocess.audio import FfmpegAudio, NoopAudio, create_audio
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.stt import SAMPLE_RATE, read_wav
from tests.media import make_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_noop_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(MediaToolError) as caught:
        NoopAudio().extract(tmp_path / "x")

    assert caught.value.code == "unavailable"


@needs_ffmpeg
def test_ffmpeg_writes_a_16khz_mono_wav(tmp_path: Path, app_config: AppConfig) -> None:
    clip = make_clip(tmp_path)
    engine = create_audio(app_config.thresholds.media.stt)

    wav = engine.extract(clip)

    assert wav == tmp_path / "audio.wav"
    assert len(read_wav(wav)) == pytest.approx(3 * SAMPLE_RATE, rel=0.1)


@needs_ffmpeg
def test_a_silent_video_has_no_audio(tmp_path: Path, app_config: AppConfig) -> None:
    clip = make_clip(tmp_path, audio=False)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None

    with pytest.raises(MediaToolError) as caught:
        FfmpegAudio(app_config.thresholds.media.stt, ffmpeg).extract(clip)

    assert caught.value.code == "no_audio"


def test_create_without_ffmpeg_is_a_noop(
    monkeypatch: pytest.MonkeyPatch, app_config: AppConfig
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    assert isinstance(create_audio(app_config.thresholds.media.stt), NoopAudio)
