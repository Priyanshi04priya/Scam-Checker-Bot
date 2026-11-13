"""Read access to the files the worker downloaded, by media key and file name."""

from pathlib import Path

from scambot.core.media_store import MediaStore


class MediaMissing(Exception):  # noqa: N818 - the message is a code, not an error text
    """The file is not there or is too large; the message is a short code."""


class MediaFiles:
    def __init__(self, store: MediaStore) -> None:
        self._store = store

    def path(self, key: str, name: str) -> Path:
        """Validated path; ``ValueError`` for a bad key or name, ``MediaMissing`` if absent."""
        try:
            return self._store.path(key, name)
        except FileNotFoundError as exc:
            msg = "media_missing"
            raise MediaMissing(msg) from exc

    def read(self, key: str, name: str, *, max_bytes: int) -> bytes:
        path = self.path(key, name)
        try:
            with path.open("rb") as handle:
                data = handle.read(max_bytes + 1)
        except OSError as exc:  # missing file, or a symlink refused by the store layout
            msg = "media_missing"
            raise MediaMissing(msg) from exc
        if len(data) > max_bytes:
            msg = "too_large"
            raise MediaMissing(msg)
        return data
