"""Tests for file validation."""

import os
from pathlib import Path

import pytest
from PIL import Image

from scambot.core.contracts import JobKind
from scambot.core.media import MAX_MEDIA_BYTES, MediaRejected
from scambot.preprocess.validate import check_file
from tests.media import EXE_BYTES, MP4_BYTES, OGG_BYTES, jpeg_bytes, png_bytes

MAX_PIXELS = 50_000_000


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def reason(path: Path, kind: JobKind) -> str:
    with pytest.raises(MediaRejected) as raised:
        check_file(path, kind, max_image_pixels=MAX_PIXELS)
    return raised.value.reason


@pytest.mark.parametrize(
    ("data", "kind", "mime"),
    [
        (png_bytes(), "image", "image/png"),
        (jpeg_bytes(), "image", "image/jpeg"),
        (OGG_BYTES, "voice", "audio/ogg"),
        (OGG_BYTES, "audio", "audio/ogg"),
        (MP4_BYTES, "video", "video/mp4"),
        (MP4_BYTES, "voice", "video/mp4"),
    ],
)
def test_real_files_are_accepted(tmp_path: Path, data: bytes, kind: JobKind, mime: str) -> None:
    info = check_file(write(tmp_path, "original", data), kind, max_image_pixels=MAX_PIXELS)

    assert info.mime_type == mime
    assert info.size_bytes == len(data)


def test_executable_renamed_to_jpg_is_refused(tmp_path: Path) -> None:
    assert reason(write(tmp_path, "photo.jpg", EXE_BYTES), "image") == "unsupported_type"


def test_text_and_wrong_kind_are_refused(tmp_path: Path) -> None:
    assert reason(write(tmp_path, "a", b"hello there"), "image") == "unsupported_type"
    assert reason(write(tmp_path, "b", MP4_BYTES), "image") == "unsupported_type"


def test_oversized_file_is_refused_without_reading_it(tmp_path: Path) -> None:
    path = tmp_path / "big"
    with path.open("wb") as handle:
        handle.truncate(MAX_MEDIA_BYTES + 1)

    assert reason(path, "video") == "too_large"


def test_empty_file_is_unreadable(tmp_path: Path) -> None:
    assert reason(write(tmp_path, "e", b""), "image") == "unreadable"


def test_png_magic_with_garbage_is_unreadable(tmp_path: Path) -> None:
    data = png_bytes()[:40] + b"garbage" * 20
    assert reason(write(tmp_path, "g", data), "image") == "unreadable"


def test_huge_image_is_refused_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "huge.png"
    Image.new("1", (8000, 8000)).save(path)

    assert reason(path, "image") == "too_large"


def test_symlink_is_unreadable(tmp_path: Path) -> None:
    real = write(tmp_path, "real", png_bytes())
    link = tmp_path / "link"
    os.symlink(real, link)

    assert reason(link, "image") == "unreadable"


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    assert reason(tmp_path / "nope", "image") == "unreadable"
