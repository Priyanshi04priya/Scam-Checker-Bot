"""Speech to text with faster-whisper, on the densest minute of speech."""

import importlib
import importlib.util
import shutil
import threading
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from scambot.core.config import SttThresholds
from scambot.core.logging import get_logger
from scambot.core.media_store import AUDIO_FILE
from scambot.core.settings import Settings
from scambot.preprocess.base import FFMPEG_INPUT_FLAGS, Extracted, MediaToolError, run_tool

_log = get_logger(component="stt")
SAMPLE_RATE: Final = 16000

Audio = NDArray[np.float32]
Span = tuple[int, int]
Vad = Callable[[Audio], Sequence[Span]]
ModelLoader = Callable[[], Any]


class SttEngine(Protocol):
    def extract(self, path: Path) -> Extracted:
        """Transcribe the speech in the audio or video file at ``path`` (blocking)."""
        ...


class NoopStt:
    def extract(self, path: Path) -> Extracted:
        raise MediaToolError("unavailable")


def to_wav(ffmpeg: str, src: Path, dst: Path, cfg: SttThresholds) -> None:
    """Convert the first audio stream of ``src`` to 16 kHz mono 16-bit WAV."""
    run_tool(
        [
            ffmpeg,
            *FFMPEG_INPUT_FLAGS,
            "-t",
            str(cfg.max_input_s),
            "-i",
            str(src),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(dst),
        ],
        cfg.ffmpeg_timeout_s,
    )


def read_wav(path: Path) -> Audio:
    """Samples of a 16 kHz mono 16-bit WAV as floats in [-1, 1]."""
    try:
        with wave.open(str(path), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (
                SAMPLE_RATE,
                1,
                2,
            ):
                raise MediaToolError("failed")
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError):
        raise MediaToolError("failed") from None
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def densest_window(spans: Sequence[Span], total: int, window: int) -> Span:
    """The ``window``-sample stretch of a ``total``-sample recording holding the most speech."""
    if total <= window:
        return (0, total)
    last_start = total - window
    best_start, best_speech = 0, -1
    for start in sorted({0, *(min(span[0], last_start) for span in spans)}):
        end = start + window
        speech = sum(max(0, min(e, end) - max(s, start)) for s, e in spans)
        if speech > best_speech:
            best_start, best_speech = start, speech
    return (best_start, best_start + window)


def silero_vad(min_silence_ms: int) -> Vad:
    """Speech spans found by the voice-activity model bundled with faster-whisper."""

    def detect(audio: Audio) -> list[Span]:
        vad = importlib.import_module("faster_whisper.vad")
        options = vad.VadOptions(min_silence_duration_ms=min_silence_ms)
        found = vad.get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
        return [(int(item["start"]), int(item["end"])) for item in found]

    return detect


class FasterWhisperStt:
    """Local CPU transcription; the model loads on first use."""

    def __init__(  # noqa: PLR0913
        self,
        cfg: SttThresholds,
        *,
        ffmpeg: str,
        model_name: str,
        compute_type: str,
        cpu_threads: int,
        vad: Vad | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._cfg = cfg
        self._ffmpeg = ffmpeg
        self._vad = vad or silero_vad(cfg.min_silence_ms)
        self._loader = model_loader or (
            lambda: importlib.import_module("faster_whisper").WhisperModel(
                model_name, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
            )
        )
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._loader()
            return self._model

    def warm_up(self) -> None:
        """Load the model now, so the first job does not wait for it."""
        self._load()

    def extract(self, path: Path) -> Extracted:
        wav = path.with_name(AUDIO_FILE)
        if not wav.exists():
            to_wav(self._ffmpeg, path, wav, self._cfg)
        audio = read_wav(wav)
        spans = self._vad(audio)
        if not spans:
            return Extracted("", None)
        start, end = densest_window(spans, len(audio), self._cfg.window_s * SAMPLE_RATE)
        segments, info = self._load().transcribe(
            audio[start:end],
            language=None,
            task="transcribe",
            beam_size=self._cfg.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self._cfg.min_silence_ms},
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(part for segment in segments if (part := segment.text.strip()))
        language = (
            info.language if info.language_probability >= self._cfg.min_language_prob else None
        )
        return Extracted(text, language)


def create_stt(settings: Settings, cfg: SttThresholds) -> SttEngine:
    """Faster-whisper when enabled, installed, and ffmpeg is available."""
    if settings.stt_engine != "faster_whisper":
        return NoopStt()
    if importlib.util.find_spec("faster_whisper") is None:
        _log.warning("stt_unavailable", reason="no_package")
        return NoopStt()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        _log.warning("stt_unavailable", reason="no_ffmpeg")
        return NoopStt()
    return FasterWhisperStt(
        cfg,
        ffmpeg=ffmpeg,
        model_name=settings.whisper_model,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
    )
