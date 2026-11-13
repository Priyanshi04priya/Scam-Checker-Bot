"""AI-image detection through an external analysis service.

A score is only ever reported as "likely AI-generated or altered"; a low score never says the
picture is real.
"""

import asyncio
import io
from dataclasses import dataclass
from typing import Any, Final

from PIL import Image

from scambot.core.config import DeepfakeThresholds
from scambot.core.contracts import CheckerResult, CheckerStatus, Evidence, Signal
from scambot.core.logging import get_logger
from scambot.mcp_server.clients.sightengine import ImageAnalyzer, SightengineError, SynthScores
from scambot.mcp_server.media import MediaFiles, MediaMissing

CHECKER: Final = "detect_deepfake"
MAX_FRAME_BYTES: Final = 30_000_000
_RECODE_QUALITY: Final = 85

_log = get_logger(component="deepfake")


@dataclass
class _Frame:
    name: str
    data: bytes = b""
    status: str = "pending"
    reason: str | None = None
    scores: SynthScores | None = None

    @property
    def score(self) -> float:
        values = (
            [v for v in (self.scores.ai_generated, self.scores.deepfake) if v is not None]
            if self.scores
            else []
        )
        return max(values, default=0.0)

    def details(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "ai_generated": self.scores.ai_generated if self.scores else None,
            "deepfake": self.scores.deepfake if self.scores else None,
        }


def _inspect(data: bytes, limits: DeepfakeThresholds) -> tuple[bytes, str | None]:
    """The bytes to upload, or a reason the picture cannot be judged."""
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if min(width, height) < limits.min_side_px:
            return data, "low_resolution"
        if len(data) / (width * height) < limits.min_bytes_per_pixel:
            return data, "over_compressed"
        if len(data) <= limits.max_upload_bytes:
            return data, None
        image.load()
        out = io.BytesIO()
        image.convert("RGB").save(out, format="JPEG", quality=_RECODE_QUALITY)
        return out.getvalue(), None


class DeepfakeChecker:
    def __init__(
        self, analyzer: ImageAnalyzer | None, media: MediaFiles, thresholds: DeepfakeThresholds
    ) -> None:
        self._analyzer = analyzer
        self._media = media
        self._limits = thresholds

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

    async def _prepare(self, media_key: str, name: str) -> _Frame:
        frame = _Frame(name)
        try:
            raw = self._media.read(media_key, name, max_bytes=MAX_FRAME_BYTES)
            frame.data, frame.reason = await asyncio.to_thread(_inspect, raw, self._limits)
        except MediaMissing as exc:
            frame.reason = str(exc)
            frame.status = "error"
            return frame
        except Exception:
            frame.reason = "unreadable_image"
            frame.status = "unusable"
            return frame
        frame.status = "unusable" if frame.reason else "usable"
        return frame

    async def _score(self, analyzer: ImageAnalyzer, frame: _Frame, gate: asyncio.Semaphore) -> None:
        try:
            async with gate:
                scores = await analyzer.scores(frame.data, frame.name)
        except SightengineError as exc:
            frame.status, frame.reason = "error", str(exc)
            return
        if scores.ai_generated is None and scores.deepfake is None:
            frame.status, frame.reason = "error", "no_score"  # absent scores are not a clean result
        else:
            frame.scores = scores
            frame.status = "scored"

    async def check(self, media_key: str, frames: list[str]) -> CheckerResult:
        """Score each picture (a video's key frames are several); report the highest score."""
        if self._analyzer is None:
            return self._result("error", {}, error="not_configured")
        names = frames[: self._limits.max_frames]
        if not names:
            return self._result("inconclusive", {"reason": "no_frames"})
        prepared = await asyncio.gather(*(self._prepare(media_key, n) for n in names))
        usable = [f for f in prepared if f.status == "usable"]
        timed_out = False
        gate = asyncio.Semaphore(self._limits.max_concurrency)
        try:
            async with asyncio.timeout(self._limits.timeout_s):
                await asyncio.gather(*(self._score(self._analyzer, f, gate) for f in usable))
        except TimeoutError:
            timed_out = True
            for frame in usable:
                if frame.status == "usable":
                    frame.status, frame.reason = "error", "timeout"
        details: dict[str, Any] = {"frames": [f.details() for f in prepared]}
        if usable:
            details["vendor_calls"] = {"sightengine": len(usable)}
        scored = [f for f in prepared if f.status == "scored"]
        if not scored:
            if usable:
                code = (
                    "timeout"
                    if timed_out
                    else next((f.reason for f in usable if f.reason), "request_failed")
                )
                return self._result("error", details, error=code)
            reason = next((f.reason for f in prepared if f.reason), "no_frames")
            return self._result("inconclusive", {**details, "reason": reason})
        top = max(scored, key=lambda f: f.score)
        score = top.score
        details["top_frame"] = top.name
        details["score"] = score
        if score >= self._limits.signal_min:
            where = f" (frame {prepared.index(top) + 1} of {len(names)})" if len(names) > 1 else ""
            summary = (
                f"An AI-image detector rated this image {round(score * 100)}% likely to be "
                f"AI-generated or altered{where}"
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
                    summary="No manipulation detected by the AI-image detector; "
                    "this is not proof that the image is unaltered"
                )
            ],
        )
