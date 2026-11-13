"""Channel adapter interface."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from scambot.core.contracts import Channel, Command, Job, Rejection
from scambot.core.media_store import MediaStore


class ChannelAdapter(Protocol):
    """Converts one platform's updates into jobs and commands, and sends replies back."""

    @property
    def channel(self) -> Channel: ...

    def parse_update(
        self, raw: Mapping[str, Any], *, trace_id: str
    ) -> Job | Command | Rejection | None:
        """Return the job, command or rejection in ``raw``, or None for ignored updates."""
        ...

    async def send_ack(self, job: Job, text: str) -> str:
        """Send the "checking" message for ``job`` and return its message ID."""
        ...

    async def send_verdict(self, job: Job, text: str) -> None:
        """Deliver the verdict, replacing the acknowledgement where the platform allows it."""
        ...

    async def send_text(self, chat_id: str, text: str) -> None:
        """Send a plain text message (command replies)."""
        ...

    async def download_media(
        self, job: Job, store: MediaStore, key: str, *, max_bytes: int, timeout_s: float
    ) -> Path:
        """Stream the job's file into ``store`` under ``key``.

        Raises MediaRejected when the platform refuses it or it is too large, and
        MediaUnavailable on a transient failure.
        """
        ...
