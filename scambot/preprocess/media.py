"""Turns a downloaded media file into text and fingerprints."""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from scambot.core.config import MediaThresholds
from scambot.core.contracts import JobKind, MediaFeatures
from scambot.core.logging import get_logger
from scambot.core.media import MediaRejected
from scambot.core.settings import Settings
from scambot.preprocess.audio import AudioExtractor, create_audio
from scambot.preprocess.base import MediaToolError, run_blocking
from scambot.preprocess.hashing import file_sha256, image_phash
from scambot.preprocess.keyframes import KeyframeExtractor, create_keyframes
from scambot.preprocess.language import LanguageDetector, create_language_detector
from scambot.preprocess.ocr import OcrEngine, create_ocr
from scambot.preprocess.stt import SttEngine, create_stt
from scambot.preprocess.validate import check_file

_log = get_logger(component="media")
CHECK_TIMEOUT_S = 5.0
HASH_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class MediaEngines:
    ocr: OcrEngine
    stt: SttEngine
    audio: AudioExtractor
    keyframes: KeyframeExtractor
    language: LanguageDetector
    pool: Executor
    limits: MediaThresholds


def create_engines(settings: Settings, limits: MediaThresholds) -> MediaEngines:
    """Build the extraction engines; a missing tool leaves its step switched off."""
    return MediaEngines(
        ocr=create_ocr(settings, limits.ocr),
        stt=create_stt(settings, limits.stt),
        audio=create_audio(limits.stt),
        keyframes=create_keyframes(limits.keyframes),
        language=create_language_detector(),
        pool=ThreadPoolExecutor(max_workers=limits.pool_workers, thread_name_prefix="media"),
        limits=limits,
    )


@dataclass
class _Collected:
    ocr_parts: list[str] = field(default_factory=list)
    transcript: str = ""
    speech_language: str | None = None
    phashes: list[str] = field(default_factory=list)
    keyframe_phashes: list[str] = field(default_factory=list)
    keyframe_names: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_lines(self, text: str) -> None:
        seen = {line for part in self.ocr_parts for line in part.splitlines()}
        fresh = [line for line in text.splitlines() if line and line not in seen]
        if fresh:
            self.ocr_parts.append("\n".join(fresh))


async def _attempt[T](
    engines: MediaEngines, found: _Collected, stage: str, fn: Callable[[], T], timeout_s: float
) -> T | None:
    """Run one blocking step; any failure is recorded and never propagates."""
    try:
        return await run_blocking(engines.pool, fn, timeout_s)
    except MediaToolError as exc:
        found.errors.append(f"{stage}:{exc.code}")
        if exc.code not in {"no_audio", "unavailable"}:
            _log.warning("media_stage_failed", stage=stage, error=exc.code)
    except TimeoutError:
        found.errors.append(f"{stage}:timeout")
    except Exception as exc:
        found.errors.append(f"{stage}:failed")
        _log.warning("media_stage_failed", stage=stage, error=type(exc).__name__)
    return None


async def _phash(engines: MediaEngines, found: _Collected, stage: str, path: Path) -> str | None:
    return await _attempt(engines, found, stage, lambda: image_phash(path), HASH_TIMEOUT_S)


async def _speech(engines: MediaEngines, found: _Collected, path: Path) -> None:
    await _attempt(
        engines,
        found,
        "audio",
        lambda: engines.audio.extract(path),
        engines.limits.stt.ffmpeg_timeout_s,
    )
    result = await _attempt(
        engines, found, "stt", lambda: engines.stt.extract(path), engines.limits.stt.timeout_s
    )
    if result is not None:
        found.transcript = result.text
        found.speech_language = result.language


async def _image(engines: MediaEngines, found: _Collected, path: Path) -> None:
    value = await _phash(engines, found, "phash", path)
    if value is not None:
        found.phashes.append(value)
    result = await _attempt(
        engines, found, "ocr", lambda: engines.ocr.extract(path), engines.limits.ocr.timeout_s
    )
    if result is not None:
        found.add_lines(result.text)


async def _frames(engines: MediaEngines, found: _Collected, path: Path) -> None:
    limits = engines.limits
    frames = await _attempt(
        engines,
        found,
        "keyframes",
        lambda: engines.keyframes.extract(path, limits.keyframes.max_keyframes),
        limits.keyframes.timeout_s,
    )
    found.keyframe_names = [frame.name for frame in frames or []]
    for frame in frames or []:
        value = await _phash(engines, found, "phash", frame)
        if value is not None:
            found.keyframe_phashes.append(value)
    for frame in (frames or [])[: limits.ocr.max_video_frames]:
        result = await _attempt(
            engines,
            found,
            "ocr",
            partial(engines.ocr.extract, frame),
            limits.ocr.timeout_s,
        )
        if result is not None:
            found.add_lines(result.text)


async def _video(engines: MediaEngines, found: _Collected, path: Path) -> None:
    async with asyncio.TaskGroup() as group:
        group.create_task(_speech(engines, found, path))
        group.create_task(_frames(engines, found, path))


async def extract_media(kind: JobKind, path: Path, engines: MediaEngines) -> MediaFeatures:
    """Validate the file and read its text. Raises MediaRejected for an unusable file."""
    started = time.perf_counter()
    limits = engines.limits
    try:
        info = await run_blocking(
            engines.pool,
            lambda: check_file(path, kind, max_image_pixels=limits.max_image_pixels),
            CHECK_TIMEOUT_S,
        )
        digest = await run_blocking(engines.pool, lambda: file_sha256(path), HASH_TIMEOUT_S)
    except TimeoutError:
        # A bare TimeoutError would be taken for the job-wide deadline by the pipeline.
        raise MediaRejected("unreadable") from None
    found = _Collected()
    try:
        async with asyncio.timeout(limits.preprocess_timeout_s) as budget:
            if kind == "image":
                await _image(engines, found, path)
            elif kind == "video":
                await _video(engines, found, path)
            else:
                await _speech(engines, found, path)
    except TimeoutError:
        if not budget.expired():
            raise
        found.errors.append("preprocess:timeout")
    all_hashes = found.phashes or found.keyframe_phashes
    features = MediaFeatures(
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
        sha256=digest,
        phash=all_hashes[0] if all_hashes else None,
        keyframe_phashes=found.keyframe_phashes,
        keyframe_names=found.keyframe_names,
        ocr_text="\n".join(found.ocr_parts),
        transcript=found.transcript,
        speech_language=found.speech_language,
        errors=found.errors,
    )
    _log.info(
        "media_ready",
        kind=kind,
        mime=features.mime_type,
        size_bytes=features.size_bytes,
        ocr_chars=len(features.ocr_text),
        transcript_chars=len(features.transcript),
        keyframes=len(features.keyframe_phashes),
        errors=features.errors,
        duration_ms=round((time.perf_counter() - started) * 1000),
    )
    return features


def compose_text(caption: str, features: MediaFeatures | None) -> str:
    """The caption, text read from the image and speech, as one text with blank lines between."""
    parts = [
        caption,
        features.ocr_text if features else "",
        features.transcript if features else "",
    ]
    return "\n\n".join(part for part in parts if part.strip())


async def detect_language(text: str, fallback: str | None, engines: MediaEngines) -> str | None:
    """Language of ``text``; ``fallback`` when there is no text or detection fails."""
    if not text.strip():
        return fallback
    try:
        found = await run_blocking(
            engines.pool, lambda: engines.language.detect(text), engines.limits.language_timeout_s
        )
    except Exception as exc:
        _log.warning("language_failed", error=type(exc).__name__)
        return fallback
    return found or fallback
