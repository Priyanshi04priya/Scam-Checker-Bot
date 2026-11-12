"""In-memory test doubles shared by unit and integration tests."""

import itertools
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import SecretStr

from scambot.core.contracts import Channel, Command, Job
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER

_COMMAND_RE = re.compile(r"/([A-Za-z0-9_]{1,32})(?:\s|$)")


def fake_update(
    update_id: int, text: str, *, user_id: int = 7, lang: str | None = "en"
) -> dict[str, Any]:
    """A raw update in the fake channel's own minimal format."""
    return {"update_id": update_id, "user_id": user_id, "text": text, "lang": lang}


@dataclass
class FakeAdapter:
    """Channel adapter that keeps every sent message in memory and supports editing."""

    channel: Channel = "telegram"
    messages: dict[str, str] = field(default_factory=dict)
    log: list[tuple[str, str, str]] = field(default_factory=list)
    fail_ack: bool = False
    _ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))

    def parse_update(self, raw: Mapping[str, Any], *, trace_id: str) -> Job | Command | None:
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            return None
        update_id = f"{self.channel}:{raw['update_id']}"
        chat_id = str(raw["user_id"])
        hashed = user_hash(raw["user_id"], SecretStr(DEV_USER_ID_PEPPER))
        # Same rule as the Telegram adapter: a name must follow the "/" directly.
        if match := _COMMAND_RE.match(text):
            return Command(
                command_id=update_id,
                trace_id=trace_id,
                channel=self.channel,
                chat_id=chat_id,
                user_hash=hashed,
                name=match.group(1).lower(),
                preferred_lang=raw.get("lang"),
            )
        return Job(
            job_id=update_id,
            trace_id=trace_id,
            channel=self.channel,
            chat_id=chat_id,
            user_hash=hashed,
            kind="text",
            text=text,
            preferred_lang=raw.get("lang"),
            received_at=datetime.now(UTC),
        )

    def _new_message(self, action: str, chat_id: str, text: str) -> str:
        message_id = str(next(self._ids))
        self.messages[message_id] = text
        self.log.append((action, chat_id, text))
        return message_id

    async def send_ack(self, job: Job, text: str) -> str:
        if self.fail_ack:
            msg = "platform unavailable"
            raise ConnectionError(msg)
        return self._new_message("ack", job.chat_id, text)

    async def send_verdict(self, job: Job, text: str) -> None:
        if job.ack_message_id in self.messages:
            self.messages[job.ack_message_id] = text
            self.log.append(("edit", job.chat_id, text))
        else:
            self._new_message("verdict", job.chat_id, text)

    async def send_text(self, chat_id: str, text: str) -> None:
        self._new_message("text", chat_id, text)
