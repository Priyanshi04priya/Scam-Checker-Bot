"""Channel adapter interface."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from scambot.core.contracts import Callback, CallbackAction, Channel, Command, Job, Rejection
from scambot.core.media_store import MediaStore


@dataclass(frozen=True)
class Button:
    """An inline button: its label and the data sent back when it is pressed."""

    label: str
    data: str


# "d:<request id>" (details), "h:" (helpful), "w:" (wrong) and "l:<language code>".
_CALLBACK_RE: Final = re.compile(r"^([dhwl]):([0-9a-f-]{36}|[A-Za-z-]{2,8})$")
_CALLBACK_ACTIONS: Final[dict[str, CallbackAction]] = {
    "d": "details",
    "h": "helpful",
    "w": "wrong",
    "l": "language",
}
_UUID_RE: Final = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def parse_callback_data(data: str) -> tuple[CallbackAction, str] | None:
    """Split button data into its action and value; None when it is not data this bot sent."""
    match = _CALLBACK_RE.match(data)
    if match is None:
        return None
    action = _CALLBACK_ACTIONS[match.group(1)]
    value = match.group(2)
    if (action == "language") == bool(_UUID_RE.match(value)):
        return None
    return action, value


class ChannelAdapter(Protocol):
    """Converts one platform's updates into jobs and commands, and sends replies back."""

    @property
    def channel(self) -> Channel: ...

    def parse_update(
        self, raw: Mapping[str, Any], *, trace_id: str
    ) -> Job | Command | Rejection | Callback | None:
        """Return the job, command, rejection or button press in ``raw``, or None if ignored."""
        ...

    async def send_ack(self, job: Job, text: str) -> str:
        """Send the "checking" message for ``job`` and return its message ID."""
        ...

    async def send_verdict(self, job: Job, text: str, *, buttons: Sequence[Button] = ()) -> None:
        """Deliver the verdict, replacing the acknowledgement where the platform allows it.

        ``buttons`` are shown in one row under the message.
        """
        ...

    async def send_text(
        self, chat_id: str, text: str, *, buttons: Sequence[Sequence[Button]] = ()
    ) -> None:
        """Send a plain text message (command replies); ``buttons`` are rows under it."""
        ...

    async def answer_callback(self, callback: Callback, text: str | None = None) -> None:
        """Acknowledge a button press, optionally with a short notice shown to the user."""
        ...

    async def download_media(
        self, job: Job, store: MediaStore, key: str, *, max_bytes: int, timeout_s: float
    ) -> Path:
        """Stream the job's file into ``store`` under ``key``.

        Raises MediaRejected when the platform refuses it or it is too large, and
        MediaUnavailable on a transient failure.
        """
        ...
