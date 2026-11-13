"""Representative frames from a video: one per scene, or evenly spaced when there is no cut."""

import importlib
import importlib.util
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from scambot.core.config import KeyframeThresholds
from scambot.core.logging import get_logger
from scambot.preprocess.base import FFMPEG_INPUT_FLAGS, MediaToolError, run_tool

_log = get_logger(component="keyframes")
MIN_SCENE_FRAMES = 15


class KeyframeExtractor(Protocol):
    def extract(self, path: Path, max_frames: int) -> list[Path]:
        """Write up to ``max_frames`` JPEG frames next to ``path`` and return them (blocking)."""
        ...


class NoopKeyframes:
    def extract(self, path: Path, max_frames: int) -> list[Path]:
        raise MediaToolError("unavailable")


def pick_times(
    scenes: Sequence[tuple[float, float]],
    duration: float | None,
    max_frames: int,
    fallback_frames: int,
) -> list[float]:
    """Seconds at which to grab frames."""
    if len(scenes) >= 2:
        mids = [(start + end) / 2 for start, end in scenes]
        if len(mids) > max_frames:
            if max_frames == 1:
                return [mids[0]]
            count = len(mids)
            picks = sorted({round(i * (count - 1) / (max_frames - 1)) for i in range(max_frames)})
            return [mids[i] for i in picks]
        return mids
    length = duration
    if length is None and scenes:
        length = scenes[0][1]
    if length is None or length <= 0:
        return [0.0]
    count = min(fallback_frames, max_frames)
    return [length * (i + 0.5) / count for i in range(count)]


class SceneKeyframes:
    def __init__(
        self, cfg: KeyframeThresholds, *, ffmpeg: str, ffprobe: str, use_scenes: bool
    ) -> None:
        self._cfg = cfg
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._use_scenes = use_scenes

    def _duration(self, path: Path) -> float | None:
        try:
            out = run_tool(
                [
                    self._ffprobe,
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                self._cfg.ffmpeg_timeout_s,
            )
            return float(out.decode("ascii", "ignore").strip())
        except (MediaToolError, ValueError):
            return None

    def _scenes(self, path: Path) -> list[tuple[float, float]]:
        if not self._use_scenes:
            return []
        try:
            scenedetect = importlib.import_module("scenedetect")
            found = scenedetect.detect(
                str(path),
                scenedetect.ContentDetector(
                    threshold=self._cfg.scene_threshold, min_scene_len=MIN_SCENE_FRAMES
                ),
                show_progress=False,
                end_time=float(self._cfg.scan_limit_s),
            )
            return [(float(start.seconds), float(end.seconds)) for start, end in found]
        except Exception as exc:  # any decoder problem just means "no cuts found"
            _log.info("scene_detect_failed", error=type(exc).__name__)
            return []

    def extract(self, path: Path, max_frames: int) -> list[Path]:
        limit = min(max_frames, self._cfg.max_keyframes)
        times = pick_times(
            self._scenes(path), self._duration(path), limit, self._cfg.fallback_frames
        )
        frames: list[Path] = []
        for i, moment in enumerate(times):
            target = path.with_name(f"kf_{i:02d}.jpg")
            try:
                run_tool(
                    [
                        self._ffmpeg,
                        *FFMPEG_INPUT_FLAGS,
                        "-ss",
                        f"{moment:.3f}",
                        "-i",
                        str(path),
                        "-frames:v",
                        "1",
                        "-vf",
                        f"scale='min({self._cfg.max_width_px},iw)':-2",
                        "-q:v",
                        "3",
                        str(target),
                    ],
                    self._cfg.ffmpeg_timeout_s,
                )
            except MediaToolError:
                continue
            if target.exists():
                frames.append(target)
        return frames


def create_keyframes(cfg: KeyframeThresholds) -> KeyframeExtractor:
    """Scene-based when scenedetect is installed; needs ffmpeg and ffprobe either way."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        _log.warning("keyframes_unavailable", reason="no_ffmpeg")
        return NoopKeyframes()
    use_scenes = importlib.util.find_spec("scenedetect") is not None
    return SceneKeyframes(cfg, ffmpeg=ffmpeg, ffprobe=ffprobe, use_scenes=use_scenes)
