"""Synthetic-voice detection.

A high score is only ever reported as "may be synthetic or cloned"; a low score never says the
voice is genuine.
"""

import asyncio
from collections.abc import Sequence
from typing import Any, Final

from scambot.core.config import VoiceThresholds
from scambot.core.contracts import CheckerResult, CheckerStatus, Evidence, Signal
from scambot.core.logging import get_logger
from scambot.mcp_server.clients.voice_spoof import VoiceSpoofDetector, VoiceSpoofError
from scambot.mcp_server.media import MediaFiles, MediaMissing
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.stt import (
    SAMPLE_RATE,
    Audio,
    Span,
    Vad,
    densest_window,
    read_wav,
    silero_vad,
)

__all__ = ["VoiceCloneChecker", "select_segment", "silero_vad"]

CHECKER: Final = "detect_voice_clone"
_log = get_logger(component="voice_clone")


def select_segment(
    audio: Audio, spans: Sequence[Span], cfg: VoiceThresholds
) -> tuple[Audio, float]:
    """The stretch of at most ``window_s`` with the most speech, and the seconds of speech in it."""
    start, end = densest_window(spans, len(audio), cfg.window_s * SAMPLE_RATE)
    speech = sum(max(0, min(e, end) - max(s, start)) for s, e in spans)
    return audio[start:end], speech / SAMPLE_RATE


class VoiceCloneChecker:
    def __init__(
        self,
        detector: VoiceSpoofDetector | None,
        media: MediaFiles,
        cfg: VoiceThresholds,
        vad: Vad | None = None,
    ) -> None:
        self._detector = detector
        self._media = media
        self._cfg = cfg
        self._vad = vad

    @property
    def detector(self) -> VoiceSpoofDetector | None:
        return self._detector

    def without_detector(self) -> "VoiceCloneChecker":
        return VoiceCloneChecker(None, self._media, self._cfg, self._vad)

    @staticmethod
    def _result(  # noqa: PLR0913
        status: CheckerStatus,
        details: dict[str, Any],
        *,
        signal: Signal = "none",
        confidence: float = 0.0,
        evidence: list[Evidence] | None = None,
        error: str | None = None,
    ) -> CheckerResult:
        return CheckerResult(
            checker=CHECKER,
            status=status,
            signal=signal,
            confidence=confidence,
            evidence=evidence or [],
            duration_ms=0,
            error=error,
            details=details,
        )

    async def _load(self, media_key: str, audio_file: str) -> Audio | CheckerResult:
        try:
            path = self._media.path(media_key, audio_file)
            size = path.stat().st_size
        except (MediaMissing, OSError, ValueError):
            return self._result("inconclusive", {"reason": "no_audio"})
        if size > self._cfg.max_wav_bytes:
            return self._result("error", {}, error="too_large")
        try:
            return await asyncio.to_thread(read_wav, path)
        except MediaToolError:
            return self._result("error", {}, error="unreadable_audio")

    async def _run(  # noqa: PLR0911
        self, detector: VoiceSpoofDetector, media_key: str, audio_file: str
    ) -> CheckerResult:
        cfg = self._cfg
        loaded = await self._load(media_key, audio_file)
        if isinstance(loaded, CheckerResult):
            return loaded
        audio = loaded
        vad = self._vad or silero_vad(cfg.min_silence_ms)
        try:
            spans = await asyncio.to_thread(vad, audio)
        except ImportError:
            return self._result("error", {}, error="not_configured")
        if not spans:
            return self._result("inconclusive", {"reason": "no_speech"})
        segment, speech_s = select_segment(audio, spans, cfg)
        if speech_s < cfg.min_speech_s:
            return self._result(
                "inconclusive", {"reason": "too_little_speech", "speech_s": round(speech_s, 2)}
            )
        paid: dict[str, Any] = (
            {"vendor_calls": {"reality_defender": 1}} if detector.name == "reality_defender" else {}
        )
        try:
            raw_score = await asyncio.to_thread(detector.score, segment)
        except VoiceSpoofError as exc:
            return self._result("error", {"detector": detector.name, **paid}, error=str(exc))
        score = min(1.0, max(0.0, float(raw_score)))
        details: dict[str, Any] = {
            "speech_s": round(speech_s, 2),
            "analysed_s": round(len(segment) / SAMPLE_RATE, 2),
            "score": score,
            "detector": detector.name,
            **paid,
        }
        if score >= cfg.signal_min:
            summary = (
                f"A voice analysis rated this voice {round(score * 100)}% likely to be "
                "synthetic or cloned; this is an uncertain signal"
            )
            return self._result(
                "ok",
                details,
                signal="likely_synthetic",
                confidence=score,
                evidence=[Evidence(summary=summary)],
            )
        return self._result(
            "ok",
            details,
            confidence=1.0 - score,
            evidence=[
                Evidence(
                    summary="No sign of a synthetic voice was detected; "
                    "this does not show the voice is genuine"
                )
            ],
        )

    async def check(self, media_key: str, audio_file: str) -> CheckerResult:
        """Rate how likely the speech in ``audio_file`` is a synthetic or cloned voice."""
        if self._detector is None:
            return self._result(
                "error", {"reason": "voice detector not installed"}, error="not_configured"
            )
        try:
            async with asyncio.timeout(self._cfg.timeout_s):
                return await self._run(self._detector, media_key, audio_file)
        except TimeoutError:
            return self._result("error", {}, error="timeout")
