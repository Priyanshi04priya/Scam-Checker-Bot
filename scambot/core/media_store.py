"""Temporary storage for downloaded media files."""

import datetime as dt
import hashlib
import os
import re
import shutil
import stat
import time
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Final, Protocol

from scambot.core.logging import get_logger
from scambot.core.media import MediaRejected

ORIGINAL: Final = "original"  # downloaded file name; no extension, the contents decide the type

_KEY_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}$")

_log = get_logger(component="media_store")


def media_key(job_id: str) -> str:
    """A directory key derived from the job ID: stable across retries, free of user content."""
    return hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:32]


class MediaStore(Protocol):
    async def put(
        self, key: str, name: str, chunks: AsyncIterable[bytes], *, max_bytes: int
    ) -> Path:
        """Write ``chunks`` to ``<key>/<name>``; MediaRejected("too_large") past ``max_bytes``."""
        ...

    def path(self, key: str, name: str) -> Path:
        """Path of ``name`` inside the existing directory for ``key``."""
        ...

    def delete(self, key: str) -> None:
        """Remove everything stored under ``key``; safe to repeat."""
        ...

    def sweep(self, older_than: dt.timedelta, *, now: float | None = None) -> int:
        """Remove entries not touched for ``older_than``; returns how many were removed."""
        ...


class LocalMediaStore:
    """Files on local disk as ``root/<key>/<name>``."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root = root.resolve()

    def _dir(self, key: str) -> Path:
        if not _KEY_RE.match(key):
            msg = "invalid media key"
            raise ValueError(msg)
        directory = self._root / key
        try:
            if stat.S_ISLNK(directory.lstat().st_mode):
                msg = "media directory is a symlink"
                raise ValueError(msg)
        except FileNotFoundError:
            pass
        if not directory.resolve().is_relative_to(self._root):
            msg = "media directory escapes the root"
            raise ValueError(msg)
        return directory

    @staticmethod
    def _check_name(name: str) -> None:
        if not _NAME_RE.match(name):
            msg = "invalid media file name"
            raise ValueError(msg)

    async def put(
        self, key: str, name: str, chunks: AsyncIterable[bytes], *, max_bytes: int
    ) -> Path:
        self._check_name(name)
        directory = self._dir(key)
        directory.mkdir(mode=0o700, exist_ok=True)
        final = directory / name
        part = directory / f"{name}.part"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        fd = os.open(part, flags, 0o600)
        try:
            total = 0
            with os.fdopen(fd, "wb") as handle:
                async for chunk in chunks:
                    total += len(chunk)
                    if total > max_bytes:
                        raise MediaRejected("too_large")
                    handle.write(chunk)
            os.replace(part, final)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        return final

    def path(self, key: str, name: str) -> Path:
        self._check_name(name)
        directory = self._dir(key)
        if not directory.is_dir():
            raise FileNotFoundError(key)
        return directory / name

    def delete(self, key: str) -> None:
        directory = self._dir(key)
        try:
            mode = directory.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISDIR(mode):
            shutil.rmtree(directory, ignore_errors=True)
        else:
            directory.unlink(missing_ok=True)

    def sweep(self, older_than: dt.timedelta, *, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - older_than.total_seconds()
        removed = 0
        for entry in os.scandir(self._root):
            try:
                if _newest_mtime(entry) >= cutoff:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.unlink(entry.path)
                removed += 1
            except OSError as exc:
                _log.warning("media_sweep_failed", error=type(exc).__name__)
        return removed


def _newest_mtime(entry: os.DirEntry[str]) -> float:
    newest = entry.stat(follow_symlinks=False).st_mtime
    if entry.is_dir(follow_symlinks=False):
        with os.scandir(entry.path) as children:
            for child in children:
                newest = max(newest, child.stat(follow_symlinks=False).st_mtime)
    return newest
