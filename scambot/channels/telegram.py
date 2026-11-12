"""Telegram adapter: update parsing, replies and a long-polling loop for local development."""

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Final

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.types import (
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    Update,
)
from pydantic import SecretStr, ValidationError

from scambot.core.contracts import Channel, Command, Job, JobKind
from scambot.core.logging import get_logger
from scambot.core.security import user_hash

CHANNEL: Final[Channel] = "telegram"
ALLOWED_UPDATES: Final = ["message"]
POLL_TIMEOUT_SECONDS: Final = 25
MAX_BACKOFF_SECONDS: Final = 30.0

# "/start", "/help@SomeBot", "/start payload": command name, optional bot mention, arguments.
_COMMAND_RE: Final = re.compile(r"^/([A-Za-z0-9_]{1,32})(?:@\w+)?(?:\s+(.*))?$", re.DOTALL)

_log = get_logger(component="telegram")

UpdateHandler = Callable[[dict[str, Any]], Awaitable[object]]


def _forward_meta(origin: object) -> dict[str, Any]:
    """Who the message was forwarded from and when, as far as Telegram reveals it."""
    match origin:
        case MessageOriginUser(date=date, sender_user=user):
            return {"type": "user", "date": date.isoformat(), "name": user.full_name}
        case MessageOriginHiddenUser(date=date, sender_user_name=name):
            return {"type": "hidden_user", "date": date.isoformat(), "name": name}
        case MessageOriginChat(date=date, sender_chat=chat):
            return {"type": "chat", "date": date.isoformat(), "name": chat.title}
        case MessageOriginChannel(date=date, chat=chat):
            return {"type": "channel", "date": date.isoformat(), "name": chat.title}
        case _:
            return {}


def create_bot(token: SecretStr, api_url: str) -> Bot:
    """Bot API client for ``token`` at ``api_url``; close it with ``bot.session.close()``."""
    session = AiohttpSession(api=TelegramAPIServer.from_base(api_url))
    return Bot(token=token.get_secret_value(), session=session)


class TelegramAdapter:
    """Telegram Bot API adapter built on aiogram."""

    def __init__(self, bot: Bot, user_id_pepper: SecretStr) -> None:
        self._bot = bot
        self._pepper = user_id_pepper

    @property
    def channel(self) -> Channel:
        return CHANNEL

    def parse_update(self, raw: Mapping[str, Any], *, trace_id: str) -> Job | Command | None:
        """Turn a webhook or polling update into a job or command.

        Only new messages in private chats from real users are handled; anything else
        (edits, group messages, stickers, malformed payloads) returns None.
        """
        try:
            update = Update.model_validate(raw)
        except ValidationError:
            _log.warning("update_invalid")
            return None
        message = update.message
        if message is None or message.from_user is None or message.from_user.is_bot:
            return None
        if message.chat.type != "private":
            return None

        update_id = f"{CHANNEL}:{update.update_id}"
        chat_id = str(message.chat.id)
        hashed_user = user_hash(message.from_user.id, self._pepper)
        language = message.from_user.language_code

        # A forwarded message is content to check, even when its text looks like a command.
        if (
            message.text is not None
            and message.forward_origin is None
            and (match := _COMMAND_RE.match(message.text))
        ):
            return Command(
                command_id=update_id,
                trace_id=trace_id,
                channel=CHANNEL,
                chat_id=chat_id,
                user_hash=hashed_user,
                name=match.group(1).lower(),
                args=(match.group(2) or "").strip(),
                preferred_lang=language,
            )

        kind: JobKind
        media_ref: str | None = None
        if message.photo:
            kind, media_ref = "image", message.photo[-1].file_id
        elif message.video is not None:
            kind, media_ref = "video", message.video.file_id
        elif message.voice is not None:
            kind, media_ref = "voice", message.voice.file_id
        elif message.audio is not None:
            kind, media_ref = "audio", message.audio.file_id
        elif message.text:
            kind = "text"
        else:
            return None

        return Job(
            job_id=update_id,
            trace_id=trace_id,
            channel=CHANNEL,
            chat_id=chat_id,
            user_hash=hashed_user,
            kind=kind,
            text=message.text if kind == "text" else message.caption,
            media_ref=media_ref,
            forward_meta=_forward_meta(message.forward_origin),
            preferred_lang=language,
            received_at=datetime.now(UTC),
        )

    async def send_ack(self, job: Job, text: str) -> str:
        sent = await self._bot.send_message(chat_id=job.chat_id, text=text)
        return str(sent.message_id)

    async def send_verdict(self, job: Job, text: str) -> None:
        if job.ack_message_id is None:
            await self._bot.send_message(chat_id=job.chat_id, text=text)
            return
        try:
            await self._bot.edit_message_text(
                text=text, chat_id=job.chat_id, message_id=int(job.ack_message_id)
            )
        except TelegramBadRequest as exc:
            # A retried job edits the message to the text it already has; that is success.
            if "message is not modified" not in exc.message:
                raise

    async def send_text(self, chat_id: str, text: str) -> None:
        await self._bot.send_message(chat_id=chat_id, text=text)


async def poll_updates(
    bot: Bot,
    handler: UpdateHandler,
    *,
    stop: asyncio.Event | None = None,
    poll_timeout: int = POLL_TIMEOUT_SECONDS,
    backoff_start: float = 1.0,
) -> None:
    """Fetch updates with long polling and pass each one, as webhook-style JSON, to ``handler``.

    Removes any webhook first (Telegram refuses polling while one is set). A failing update is
    logged and skipped so one bad message cannot stall the loop. Runs until ``stop`` is set.
    """
    stop = stop or asyncio.Event()
    await bot.delete_webhook(drop_pending_updates=False)
    offset: int | None = None
    backoff = backoff_start
    while not stop.is_set():
        try:
            updates = await bot.get_updates(
                offset=offset,
                timeout=poll_timeout,
                allowed_updates=ALLOWED_UPDATES,
                request_timeout=poll_timeout + 10,
            )
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            continue
        except (TelegramNetworkError, TelegramServerError) as exc:
            _log.warning("polling_failed", error=type(exc).__name__, retry_in_s=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue
        backoff = backoff_start
        for update in updates:
            offset = update.update_id + 1
            try:
                # Only the fields Telegram sent: aiogram fills unset fields (for example in
                # link_preview_options) with placeholder objects that cannot be serialised.
                raw = update.model_dump(
                    mode="json", by_alias=True, exclude_unset=True, exclude_none=True
                )
                await handler(raw)
            except Exception:
                _log.exception("update_failed", update_id=update.update_id)


if __name__ == "__main__":
    # Local development entry point: python -m scambot.channels.telegram --poll
    from scambot.api.polling import main

    raise SystemExit(main())
