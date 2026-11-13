"""Tests for keyframe selection and extraction."""

import importlib.util
import shutil
from pathlib import Path

import pytest
from PIL import Image

from scambot.core.config import AppConfig, KeyframeThresholds
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.keyframes import NoopKeyframes, SceneKeyframes, create_keyframes, pick_times
from tests.media import make_clip

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


def test_pick_times_uses_scene_midpoints() -> None:
    assert pick_times([(0, 2), (2, 6)], 6, 6, 3) == [1.0, 4.0]


def test_pick_times_caps_the_count_evenly() -> None:
    scenes = [(float(i), float(i + 1)) for i in range(10)]

    times = pick_times(scenes, 10, 6, 3)

    assert len(times) == 6
    assert times[0] == 0.5
    assert times[-1] == 9.5
    assert times == sorted(times)
    assert pick_times(scenes, 10, 1, 3) == [0.5]


def test_pick_times_falls_back_to_even_spacing() -> None:
    assert pick_times([(0, 9)], None, 6, 3) == [1.5, 4.5, 7.5]
    assert pick_times([], 12, 6, 3) == [2.0, 6.0, 10.0]
    assert pick_times([], 12, 2, 3) == [3.0, 9.0]


def test_pick_times_with_unknown_duration() -> None:
    assert pick_times([], None, 6, 3) == [0.0]
    assert pick_times([], 0, 6, 3) == [0.0]


@pytest.fixture
def cfg(app_config: AppConfig) -> KeyframeThresholds:
    return app_config.thresholds.media.keyframes


def engine(cfg: KeyframeThresholds, *, use_scenes: bool) -> SceneKeyframes:
    return SceneKeyframes(
        cfg,
        ffmpeg=shutil.which("ffmpeg") or "ffmpeg",
        ffprobe=shutil.which("ffprobe") or "ffprobe",
        use_scenes=use_scenes,
    )


@needs_ffmpeg
def test_even_spacing_without_scene_detection(cfg: KeyframeThresholds, tmp_path: Path) -> None:
    clip = make_clip(tmp_path)

    frames = engine(cfg, use_scenes=False).extract(clip, 6)

    assert [f.name for f in frames] == ["kf_00.jpg", "kf_01.jpg", "kf_02.jpg"]
    with Image.open(frames[0]) as image:
        assert image.size == (160, 120)


@needs_ffmpeg
def test_scene_detection_gives_one_frame_per_scene(cfg: KeyframeThresholds, tmp_path: Path) -> None:
    pytest.importorskip("scenedetect")
    clip = make_clip(tmp_path)

    frames = engine(cfg, use_scenes=True).extract(clip, 6)

    assert len(frames) == 2
    with Image.open(frames[0]) as first, Image.open(frames[1]) as second:
        red = first.convert("RGB").resize((1, 1)).tobytes()
        blue = second.convert("RGB").resize((1, 1)).tobytes()
    assert red[0] > red[2]
    assert blue[2] > blue[0]


def test_unreadable_video_gives_no_frames(cfg: KeyframeThresholds, tmp_path: Path) -> None:
    broken = tmp_path / "original"
    broken.write_bytes(b"not a video")

    assert engine(cfg, use_scenes=False).extract(broken, 6) == []


def test_create_keyframes_needs_ffmpeg(
    cfg: KeyframeThresholds, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    noop = create_keyframes(cfg)
    assert isinstance(noop, NoopKeyframes)
    with pytest.raises(MediaToolError):
        noop.extract(Path("x"), 3)

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert isinstance(create_keyframes(cfg), SceneKeyframes)
