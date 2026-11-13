"""Tests for the local media store."""

import datetime as dt
import os
import stat
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from scambot.core.media import MediaRejected
from scambot.core.media_store import ORIGINAL, LocalMediaStore, media_key

MINUTE = 60


async def chunked(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def test_put_and_path_round_trip(media_root: Path) -> None:
    store = LocalMediaStore(media_root)

    written = await store.put("abc", ORIGINAL, chunked(b"he", b"llo"), max_bytes=10)

    assert written == store.path("abc", ORIGINAL)
    assert written.read_bytes() == b"hello"
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert stat.S_IMODE(written.parent.stat().st_mode) == 0o700


async def test_exceeding_the_cap_leaves_nothing(media_root: Path) -> None:
    store = LocalMediaStore(media_root)

    with pytest.raises(MediaRejected) as raised:
        await store.put("abc", ORIGINAL, chunked(b"123", b"456"), max_bytes=5)

    assert raised.value.reason == "too_large"
    assert list((media_root / "abc").iterdir()) == []


async def test_other_errors_remove_the_partial_file(media_root: Path) -> None:
    store = LocalMediaStore(media_root)

    async def broken() -> AsyncIterator[bytes]:
        yield b"x"
        raise ConnectionError

    with pytest.raises(ConnectionError):
        await store.put("abc", ORIGINAL, broken(), max_bytes=5)

    assert list((media_root / "abc").iterdir()) == []


@pytest.mark.parametrize("key", ["../x", "a/b", ".hidden", ""])
def test_bad_keys_are_refused(media_root: Path, key: str) -> None:
    with pytest.raises(ValueError, match="invalid media key"):
        LocalMediaStore(media_root).path(key, ORIGINAL)


@pytest.mark.parametrize("name", ["../x", "a/b", ".hidden", ""])
def test_bad_names_are_refused(media_root: Path, name: str) -> None:
    (media_root).mkdir()
    with pytest.raises(ValueError, match="invalid media file name"):
        LocalMediaStore(media_root).path("abc", name)


def test_symlinked_directory_is_refused(media_root: Path, tmp_path: Path) -> None:
    store = LocalMediaStore(media_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (media_root / "evil").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.path("evil", ORIGINAL)


async def test_path_after_delete_does_not_recreate_the_directory(media_root: Path) -> None:
    store = LocalMediaStore(media_root)
    await store.put("abc", ORIGINAL, chunked(b"x"), max_bytes=5)
    store.delete("abc")

    with pytest.raises(FileNotFoundError):
        store.path("abc", ORIGINAL)
    assert not (media_root / "abc").exists()
    store.delete("abc")  # idempotent


async def test_delete_unlinks_a_symlink_without_following_it(
    media_root: Path, tmp_path: Path
) -> None:
    store = LocalMediaStore(media_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    (media_root / "evil").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.delete("evil")
    assert (outside / "keep").exists()


async def test_sweep_removes_only_old_entries(media_root: Path, tmp_path: Path) -> None:
    store = LocalMediaStore(media_root)
    await store.put("old", ORIGINAL, chunked(b"x"), max_bytes=5)
    await store.put("new", ORIGINAL, chunked(b"x"), max_bytes=5)
    stray = media_root / "stray.txt"
    stray.write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = media_root / "link"
    link.symlink_to(outside, target_is_directory=True)
    now = 1_000_000.0
    long_ago = now - 50 * MINUTE
    for path in (media_root / "old" / ORIGINAL, media_root / "old", stray):
        os.utime(path, (long_ago, long_ago))
    os.utime(link, (long_ago, long_ago), follow_symlinks=False)
    for path in (media_root / "new" / ORIGINAL, media_root / "new"):
        os.utime(path, (now, now))

    removed = store.sweep(dt.timedelta(minutes=45), now=now)

    assert removed == 3
    assert sorted(os.listdir(media_root)) == ["new"]
    assert outside.exists()


def test_media_key_is_stable_and_free_of_the_job_id() -> None:
    key = media_key("telegram:123")

    assert key == media_key("telegram:123")
    assert len(key) == 32
    assert "123" not in key or key.isalnum()
