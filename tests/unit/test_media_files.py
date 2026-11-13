"""Tests for the checker server's read access to downloaded media."""

from pathlib import Path

import pytest

from scambot.core.media_store import ORIGINAL, LocalMediaStore, MediaStore
from scambot.mcp_server.media import MediaFiles, MediaMissing
from tests.media import store_bytes


async def put(store: MediaStore, key: str, data: bytes) -> None:
    await store_bytes(store, key, ORIGINAL, data)


async def test_reads_a_stored_file(media_store: MediaStore) -> None:
    await put(media_store, "k1", b"hello")

    assert MediaFiles(media_store).read("k1", ORIGINAL, max_bytes=10) == b"hello"


async def test_too_large(media_store: MediaStore) -> None:
    await put(media_store, "k1", b"hello")

    with pytest.raises(MediaMissing, match="too_large"):
        MediaFiles(media_store).read("k1", ORIGINAL, max_bytes=4)


async def test_missing_file_and_directory(media_store: MediaStore) -> None:
    await put(media_store, "k1", b"x")
    files = MediaFiles(media_store)

    with pytest.raises(MediaMissing, match="media_missing"):
        files.read("k1", "kf_00.jpg", max_bytes=10)
    with pytest.raises(MediaMissing, match="media_missing"):
        files.read("other", ORIGINAL, max_bytes=10)


@pytest.mark.parametrize(
    ("key", "name"),
    [("../x", ORIGINAL), ("k1", "a/b"), ("k1", "/etc/passwd"), ("/abs", ORIGINAL), ("k1", "..")],
)
async def test_traversal_is_rejected(media_store: MediaStore, key: str, name: str) -> None:
    await put(media_store, "k1", b"x")

    with pytest.raises(ValueError, match="invalid"):
        MediaFiles(media_store).read(key, name, max_bytes=10)


async def test_symlinked_directory_is_rejected(media_root: Path, tmp_path: Path) -> None:
    store = LocalMediaStore(media_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ORIGINAL).write_bytes(b"secret")
    (media_root / "link").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        MediaFiles(store).read("link", ORIGINAL, max_bytes=10)
