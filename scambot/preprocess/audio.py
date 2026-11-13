"""Extracting a clip's sound track as a 16 kHz mono WAV for the voice check."""

import shutil
from pathlib import Path
from typing import Protocol

from scambot.core.config import SttThresholds
from scambot.core.logging import get_logger
from scambot.core.media_store import AUDIO_FILE
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.stt import to_wav

__all__ = ["AUDIO_FILE", "AudioExtractor", "FfmpegAudio", "NoopAudio", "create_audio"]

_log = get_logger(component="audio")


class AudioExtractor(Protocol):
    def extract(self, path: Path) -> Path:
        """Write the sound track beside ``path`` as ``audio.wav`` (blocking)."""
        ...


class NoopAudio:
    def extract(self, path: Path) -> Path:
        raise MediaToolError("unavailable")


class FfmpegAudio:
    def __init__(self, cfg: SttThresholds, ffmpeg: str) -> None:
        self._cfg = cfg
        self._ffmpeg = ffmpeg

    def extract(self, path: Path) -> Path:
        wav = path.with_name(AUDIO_FILE)
        try:
            to_wav(self._ffmpeg, path, wav, self._cfg)
        except BaseException:
            wav.unlink(missing_ok=True)  # a partial file must not be mistaken for the sound track
            raise
        return wav


def create_audio(cfg: SttThresholds) -> AudioExtractor:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        _log.warning("audio_unavailable", reason="no_ffmpeg")
        return NoopAudio()
    return FfmpegAudio(cfg, ffmpeg)
