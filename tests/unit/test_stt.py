"""Tests for speech-to-text."""

import importlib.util
import shutil
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scambot.core.config import AppConfig, SttThresholds
from scambot.core.settings import Settings
from scambot.preprocess import stt as stt_module
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.stt import (
    SAMPLE_RATE,
    FasterWhisperStt,
    NoopStt,
    create_stt,
    densest_window,
    read_wav,
    to_wav,
)
from tests.media import make_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def write_wav(path: Path, samples: int, *, rate: int = SAMPLE_RATE, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x40" * samples * channels)


@pytest.fixture
def cfg(app_config: AppConfig) -> SttThresholds:
    return app_config.thresholds.media.stt


def test_densest_window() -> None:
    assert densest_window([(10, 20)], 100, 200) == (0, 100)  # short audio: all of it
    assert densest_window([(500, 600)], 1000, 200) == (500, 700)  # one span
    assert densest_window([(0, 10), (900, 1000)], 1000, 200) == (800, 1000)  # speech late
    assert densest_window([(0, 100), (500, 600)], 1000, 200) == (0, 200)  # tie: earliest
    assert densest_window([], 1000, 200) == (0, 200)


def test_read_wav_scales_to_floats(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    write_wav(path, 100)

    audio = read_wav(path)

    assert audio.dtype == np.float32
    assert audio.shape == (100,)
    assert audio[0] == pytest.approx(0.5)


@pytest.mark.parametrize(("rate", "channels"), [(44100, 1), (SAMPLE_RATE, 2)])
def test_read_wav_requires_16k_mono(tmp_path: Path, rate: int, channels: int) -> None:
    path = tmp_path / "a.wav"
    write_wav(path, 10, rate=rate, channels=channels)

    with pytest.raises(MediaToolError):
        read_wav(path)


def test_read_wav_rejects_garbage(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    path.write_bytes(b"nope")

    with pytest.raises(MediaToolError):
        read_wav(path)


def fake_engine(
    cfg: SttThresholds, tmp_path: Path, **overrides: Any
) -> tuple[FasterWhisperStt, Path, list[Any]]:
    source = tmp_path / "original"
    source.write_bytes(b"x")
    calls: list[Any] = []

    class Model:
        def transcribe(self, audio: Any, **kwargs: Any) -> Any:
            calls.append((audio, kwargs))
            segments = [
                SimpleNamespace(text=" नमस्ते "),
                SimpleNamespace(text=" "),
                SimpleNamespace(text="दोस्तों"),
            ]
            return iter(segments), SimpleNamespace(language="hi", language_probability=0.9)

    options: dict[str, Any] = {
        "vad": lambda audio: [(0, len(audio))],
        "model_loader": Model,
    } | overrides
    engine = FasterWhisperStt(
        cfg, ffmpeg="ffmpeg", model_name="small", compute_type="int8", cpu_threads=0, **options
    )
    return engine, source, calls


def stub_conversion(monkeypatch: pytest.MonkeyPatch, samples: int) -> None:
    def fake(_ffmpeg: str, _src: Path, dst: Path, _cfg: SttThresholds) -> None:
        write_wav(dst, samples)

    monkeypatch.setattr(stt_module, "to_wav", fake)


def test_transcribes_only_the_densest_window(
    cfg: SttThresholds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_conversion(monkeypatch, 90 * SAMPLE_RATE)
    engine, source, calls = fake_engine(cfg, tmp_path)

    result = engine.extract(source)

    assert result.text == "नमस्ते दोस्तों"
    assert result.language == "hi"
    ((audio, kwargs),) = calls
    assert len(audio) == 60 * SAMPLE_RATE
    assert kwargs["beam_size"] == 1
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["language"] is None


def test_unsure_language_is_dropped(
    cfg: SttThresholds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_conversion(monkeypatch, SAMPLE_RATE)

    class Unsure:
        def transcribe(self, audio: Any, **_kwargs: Any) -> Any:
            return iter([SimpleNamespace(text="hmm")]), SimpleNamespace(
                language="fr", language_probability=0.2
            )

    engine, source, _ = fake_engine(cfg, tmp_path, model_loader=Unsure)

    result = engine.extract(source)

    assert (result.text, result.language) == ("hmm", None)


def test_no_speech_never_loads_the_model(
    cfg: SttThresholds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_conversion(monkeypatch, SAMPLE_RATE)

    def loader() -> Any:
        raise AssertionError

    engine, source, _ = fake_engine(cfg, tmp_path, vad=lambda _audio: [], model_loader=loader)

    assert engine.extract(source).text == ""


def test_warm_up_loads_the_model_once(cfg: SttThresholds, tmp_path: Path) -> None:
    loads: list[int] = []

    def loader() -> Any:
        loads.append(1)
        return object()

    engine, _, _ = fake_engine(cfg, tmp_path, model_loader=loader)
    engine.warm_up()
    engine.warm_up()

    assert loads == [1]


def test_create_stt_choices(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = app_config.thresholds.media.stt
    assert isinstance(create_stt(Settings(stt_engine="none"), cfg), NoopStt)
    with pytest.raises(MediaToolError):
        NoopStt().extract(Path("x"))

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert isinstance(create_stt(Settings(), cfg), NoopStt)

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert isinstance(create_stt(Settings(), cfg), NoopStt)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    assert isinstance(create_stt(Settings(), cfg), FasterWhisperStt)


@needs_ffmpeg
def test_to_wav_gives_16k_mono(cfg: SttThresholds, tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    target = tmp_path / "audio.wav"

    to_wav(shutil.which("ffmpeg") or "ffmpeg", clip, target, cfg)

    with wave.open(str(target), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (
            16000,
            1,
            2,
        )
        assert handle.getnframes() > 16000


@needs_ffmpeg
def test_video_without_audio_is_reported(cfg: SttThresholds, tmp_path: Path) -> None:
    clip = make_clip(tmp_path, audio=False)

    with pytest.raises(MediaToolError) as raised:
        to_wav(shutil.which("ffmpeg") or "ffmpeg", clip, tmp_path / "a.wav", cfg)
    assert raised.value.code == "no_audio"
