"""Channel adapter interface."""

from collections.abc import Mapping
from typing import Any, Protocol

from scambot.core.contracts import Channel, Command, Job


class ChannelAdapter(Protocol):
    """Converts one platform's updates into jobs and commands, and sends replies back."""

    @property
    def channel(self) -> Channel: ...

    def parse_update(self, raw: Mapping[str, Any], *, trace_id: str) -> Job | Command | None:
        """Return the job or command in ``raw``, or None for updates the bot ignores."""
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
