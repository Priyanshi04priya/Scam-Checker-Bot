"""Checks on a downloaded file: size, real type from its bytes, and readability."""

from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG

import filetype
from PIL import Image, UnidentifiedImageError

from scambot.core.contracts import JobKind
from scambot.core.media import ACCEPTED_MIME, MAX_MEDIA_BYTES, MediaRejected


@dataclass(frozen=True)
class FileInfo:
    mime_type: str
    size_bytes: int


def check_file(
    path: Path, kind: JobKind, *, max_image_pixels: int, max_bytes: int = MAX_MEDIA_BYTES
) -> FileInfo:
    """Return the file's real type and size, or raise MediaRejected.

    The type comes from the file's magic bytes, never from its name or declared type.
    """
    try:
        info = path.lstat()
    except OSError:
        raise MediaRejected("unreadable") from None
    if not S_ISREG(info.st_mode) or info.st_size == 0:
        raise MediaRejected("unreadable")
    if info.st_size > max_bytes:
        raise MediaRejected("too_large")
    mime = filetype.guess_mime(str(path))
    if mime is None or mime not in ACCEPTED_MIME[kind]:
        raise MediaRejected("unsupported_type")
    if kind == "image":
        _check_image(path, max_image_pixels)
    return FileInfo(mime_type=mime, size_bytes=info.st_size)


def _check_image(path: Path, max_pixels: int) -> None:
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise MediaRejected("too_large")
            image.verify()
    except Image.DecompressionBombError:
        raise MediaRejected("too_large") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise MediaRejected("unreadable") from None
