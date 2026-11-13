"""Content fingerprints. Only hashes of messages are stored, never the text itself."""

import hashlib
import re
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

_SPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lower-case with runs of whitespace collapsed, so trivial edits hash the same."""
    return _SPACE_RE.sub(" ", text.lower()).strip()


def text_sha256(text: str) -> str | None:
    """SHA-256 of the normalised text, or None when there is no text."""
    normalised = normalise_text(text)
    if not normalised:
        return None
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def image_phash(source: Path | Image.Image) -> str:
    """Perceptual hash (64 bits, 16 hex digits) of an image, after applying its rotation."""
    if isinstance(source, Path):
        with Image.open(source) as opened:
            return str(imagehash.phash(ImageOps.exif_transpose(opened)))
    return str(imagehash.phash(ImageOps.exif_transpose(source)))


def phash_to_int(value: str) -> int:
    """The hash as a signed 64-bit integer, as the BIGINT column needs."""
    number = int(value, 16)
    return number - 2**64 if number >= 2**63 else number
