"""Tests for content fingerprints."""

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageDraw

from scambot.preprocess.hashing import (
    file_sha256,
    image_phash,
    normalise_text,
    phash_to_int,
    text_sha256,
)


def test_normalise_text() -> None:
    assert normalise_text("  Your  KYC\n\tEXPIRES  ") == "your kyc expires"


def test_trivial_edits_hash_the_same() -> None:
    assert text_sha256("Your KYC expires") == text_sha256("  your kyc\nexpires ")
    assert text_sha256("Your KYC expires") != text_sha256("Your KYC expired")
    assert len(text_sha256("बिजली") or "") == 64


def test_empty_text_has_no_hash() -> None:
    assert text_sha256("") is None
    assert text_sha256(" \n ") is None


def picture(shapes: int) -> Image.Image:
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    for i in range(shapes):
        draw.rectangle((10 + i * 35, 20 + i * 25, 50 + i * 35, 90 + i * 25), fill=(20 * i, 50, 200))
    draw.ellipse((100, 110, 180, 190), fill="black")
    return image


def distance(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc" * 500_000)

    assert file_sha256(path) == hashlib.sha256(b"abc" * 500_000).hexdigest()


def test_phash_survives_recompression(tmp_path: Path) -> None:
    original = tmp_path / "a.png"
    picture(4).save(original)
    buffer = io.BytesIO()
    picture(4).save(buffer, format="JPEG", quality=60)
    recoded = tmp_path / "a.jpg"
    recoded.write_bytes(buffer.getvalue())
    other = tmp_path / "b.png"
    Image.effect_noise((200, 200), 80).convert("RGB").save(other)

    first = image_phash(original)

    assert len(first) == 16
    assert distance(first, image_phash(recoded)) <= 8
    assert distance(first, image_phash(other)) > 8
    assert image_phash(Image.open(original)) == first


def test_phash_to_int_is_a_signed_64_bit_number() -> None:
    assert phash_to_int("ffffffffffffffff") == -1
    assert phash_to_int("0" * 16) == 0
    assert phash_to_int("8000000000000000") == -(2**63)
    assert phash_to_int("7fffffffffffffff") == 2**63 - 1
