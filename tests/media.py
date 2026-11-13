"""Byte strings and small files for media tests."""

import io
import random
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

from PIL import Image, ImageDraw

from scambot.core.media_store import MediaStore

OGG_BYTES = b"OggS" + b"\0" * 60
MP4_BYTES = b"\0\0\0\x18ftypisom" + b"\0" * 60
EXE_BYTES = b"MZ\x90\0" + b"\0" * 200
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "media"
HINDI_NOTICE = FIXTURE_DIR / "hindi_notice.png"


def png_bytes(
    size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(color: tuple[int, int, int] = (30, 30, 200), quality: int = 60) -> bytes:
    image = Image.new("RGB", (64, 48), color)
    for x in range(0, 64, 8):  # some structure so the hash is not degenerate
        for y in range(48):
            image.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def make_clip(directory: Path, *, audio: bool = True) -> Path:
    """Two coloured 1.5 s scenes (red, then blue), with a sine tone when ``audio``."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    target = directory / ("clip.mp4" if audio else "silent.mp4")
    args = [
        ffmpeg, "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:r=25:d=1.5",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=25:d=1.5",
    ]  # fmt: skip
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
    args += ["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]"]
    if audio:
        args += ["-map", "2:a", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)]
    subprocess.run(args, check=True, capture_output=True)  # noqa: S603
    return target


def is_empty(directory: Path) -> bool:
    """True when nothing is left in ``directory`` (or it does not exist)."""
    return not directory.exists() or not any(directory.iterdir())


def exif_jpeg(
    taken: str = "2019:08:10 09:30:00", make: str = "Acme", model: str = "Cam 1", size: int = 400
) -> bytes:
    """A noisy JPEG (so it is not over-compressed) with camera data in its EXIF block."""
    import random  # noqa: PLC0415

    rng = random.Random(7)  # noqa: S311 - test data
    image = Image.new("RGB", (size, size))
    image.putdata(
        [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size * size)]
    )
    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0131] = "EditorX"
    exif.get_ifd(0x8769)[36867] = taken
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, exif=exif)
    return buffer.getvalue()


async def store_bytes(store: MediaStore, key: str, name: str, data: bytes) -> None:
    """Put ``data`` in the media store as ``<key>/<name>``."""

    async def chunks() -> AsyncIterator[bytes]:
        yield data

    await store.put(key, name, chunks(), max_bytes=50_000_000)


def textured_jpeg(size: int = 480, seed: int = 3) -> bytes:
    """A detailed picture (gradients, shapes, fine grain) that has a meaningful perceptual hash."""
    rng = random.Random(seed)  # noqa: S311 - test data
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    assert pixels is not None
    for x in range(size):
        for y in range(size):
            pixels[x, y] = (
                (x * 255 // size + rng.randrange(12)) % 256,
                (y * 255 // size + rng.randrange(12)) % 256,
                ((x + y) * 255 // (2 * size) + rng.randrange(12)) % 256,
            )
    draw = ImageDraw.Draw(image)
    for _ in range(14):
        x0, y0 = rng.randrange(size - 80), rng.randrange(size - 80)
        box = (x0, y0, x0 + rng.randrange(30, 160), y0 + rng.randrange(30, 160))
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        draw.ellipse(box, fill=colour) if rng.random() < 0.5 else draw.rectangle(box, fill=colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def recompressed(data: bytes, *, quality: int = 40, scale: float = 0.9) -> bytes:
    """The same picture saved again the way a messaging app would: smaller and more compressed."""
    with Image.open(io.BytesIO(data)) as opened:
        smaller = opened.resize((round(opened.width * scale), round(opened.height * scale)))
    buffer = io.BytesIO()
    smaller.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
