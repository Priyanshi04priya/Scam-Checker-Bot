"""Telegram adapter: update parsing, replies and a long-polling loop for local development."""

import asyncio
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import aiohttp
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
    Animation,
    Audio,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    MessageOriginChannel,
    MessageOriginChat,
    MessageOriginHiddenUser,
    MessageOriginUser,
    Update,
    Video,
    Voice,
)
from pydantic import SecretStr, ValidationError

from scambot.channels.base import Button, parse_callback_data
from scambot.core.contracts import (
    Callback,
    CallbackAction,
    Channel,
    Command,
    Job,
    JobKind,
    MediaMeta,
    MediaSource,
    Rejection,
)
from scambot.core.logging import get_logger
from scambot.core.media import MediaRejected, MediaUnavailable, kind_for_mime
from scambot.core.media_store import ORIGINAL, MediaStore
from scambot.core.security import user_hash

CHANNEL: Final[Channel] = "telegram"
ALLOWED_UPDATES: Final = ["message", "callback_query"]
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


def _markup(rows: Sequence[Sequence[Button]]) -> InlineKeyboardMarkup | None:
    if not any(rows):
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=b.label, callback_data=b.data) for b in row]
            for row in rows
            if row
        ]
    )


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

    def _parse_callback(
        self, query: CallbackQuery, update_id: int, trace_id: str
    ) -> Callback | None:
        message = query.message
        if query.from_user.is_bot or message is None or message.chat.type != "private":
            return None
        if not isinstance(message, Message):
            # An old message Telegram no longer hands over: the press still gets an answer.
            return self._callback(query, message.chat.id, "expired", "-", trace_id)
        parsed = parse_callback_data(query.data or "")
        if parsed is None:
            _log.info("callback_ignored", reason="data")
            return None
        action, value = parsed
        return self._callback(query, message.chat.id, action, value, trace_id)

    def _callback(
        self, query: CallbackQuery, chat_id: int, action: CallbackAction, value: str, trace_id: str
    ) -> Callback:
        return Callback(
            callback_id=f"{CHANNEL}:cb:{query.id}",
            query_id=query.id,
            trace_id=trace_id,
            channel=CHANNEL,
            chat_id=str(chat_id),
            user_hash=user_hash(query.from_user.id, self._pepper),
            action=action,
            value=value,
            preferred_lang=query.from_user.language_code,
        )

    def parse_update(  # noqa: PLR0911
        self, raw: Mapping[str, Any], *, trace_id: str
    ) -> Job | Command | Rejection | Callback | None:
        """Turn a webhook or polling update into a job, command, rejection or button press.

        Only new messages and button presses in private chats from real users are handled;
        anything else (edits, group messages, stickers, malformed payloads) returns None.
        """
        try:
            update = Update.model_validate(raw)
        except ValidationError:
            _log.warning("update_invalid")
            return None
        if update.callback_query is not None:
            return self._parse_callback(update.callback_query, update.update_id, trace_id)
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
        media: MediaMeta | None = None
        found = self._media_of(message)
        if found is not None:
            kind_hint, media_ref, media = found
            if kind_hint is None:
                return Rejection(
                    rejection_id=update_id,
                    trace_id=trace_id,
                    channel=CHANNEL,
                    chat_id=chat_id,
                    user_hash=hashed_user,
                    reason="unsupported_type",
                    preferred_lang=language,
                )
            kind = kind_hint
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
            media=media,
            forward_meta=_forward_meta(message.forward_origin),
            preferred_lang=language,
            received_at=datetime.now(UTC),
        )

    @staticmethod
    def _media_of(message: Message) -> tuple[JobKind | None, str, MediaMeta] | None:
        """The media in ``message`` as (kind, file ID, declared metadata); kind None if unsupported.

        The order matters: Telegram also sets ``document`` on an animation.
        """
        item: Animation | Video | Voice | Audio
        source: MediaSource
        kind: JobKind
        if message.photo:
            photo = message.photo[-1]
            return "image", photo.file_id, MediaMeta(source="photo", file_size=photo.file_size)
        if message.animation is not None:
            item, source, kind = message.animation, "animation", "video"
        elif message.video is not None:
            item, source, kind = message.video, "video", "video"
        elif message.video_note is not None:
            meta = MediaMeta(
                source="video_note",
                file_size=message.video_note.file_size,
                duration_s=message.video_note.duration,
            )
            return "video", message.video_note.file_id, meta
        elif message.voice is not None:
            item, source, kind = message.voice, "voice", "voice"
        elif message.audio is not None:
            item, source, kind = message.audio, "audio", "audio"
        elif message.document is not None:
            document = message.document
            meta = MediaMeta(
                source="document", file_size=document.file_size, mime_type=document.mime_type
            )
            return kind_for_mime(document.mime_type), document.file_id, meta
        else:
            return None
        meta = MediaMeta(
            source=source,
            file_size=item.file_size,
            mime_type=item.mime_type,
            duration_s=item.duration,
        )
        return kind, item.file_id, meta

    async def download_media(
        self, job: Job, store: MediaStore, key: str, *, max_bytes: int, timeout_s: float
    ) -> Path:
        if not job.media_ref:
            raise MediaRejected("unreadable")
        timeout = math.ceil(timeout_s)
        try:
            file = await self._bot.get_file(job.media_ref, request_timeout=timeout)
        except TelegramBadRequest as exc:
            if "file is too big" in exc.message.lower():
                raise MediaRejected("too_large") from None
            raise MediaRejected("unreadable") from None
        except (TelegramNetworkError, TelegramServerError, TelegramRetryAfter) as exc:
            raise MediaUnavailable(type(exc).__name__) from None
        if file.file_size is not None and file.file_size > max_bytes:
            raise MediaRejected("too_large")
        if not file.file_path:
            raise MediaRejected("unreadable")
        # Errors raised below can carry the file URL, which contains the bot token: they are
        # never chained or logged.
        url = self._bot.session.api.file_url(self._bot.token, file.file_path)
        chunks = self._bot.session.stream_content(
            url=url, timeout=timeout, chunk_size=65536, raise_for_status=True
        )
        try:
            return await store.put(key, ORIGINAL, chunks, max_bytes=max_bytes)
        except aiohttp.ClientResponseError as exc:
            if 400 <= exc.status < 500:
                raise MediaRejected("unreadable") from None
            raise MediaUnavailable(type(exc).__name__) from None
        except (
            aiohttp.ClientError,
            TelegramNetworkError,
            TelegramServerError,
            TelegramRetryAfter,
        ) as exc:
            raise MediaUnavailable(type(exc).__name__) from None

    async def send_ack(self, job: Job, text: str) -> str:
        sent = await self._bot.send_message(chat_id=job.chat_id, text=text)
        return str(sent.message_id)

    async def send_verdict(self, job: Job, text: str, *, buttons: Sequence[Button] = ()) -> None:
        markup = _markup([buttons])
        if job.ack_message_id is None:
            await self._bot.send_message(chat_id=job.chat_id, text=text, reply_markup=markup)
            return
        try:
            await self._bot.edit_message_text(
                text=text,
                chat_id=job.chat_id,
                message_id=int(job.ack_message_id),
                reply_markup=markup,
            )
        except TelegramBadRequest as exc:
            # A retried job edits the message to the text it already has; that is success.
            if "message is not modified" not in exc.message:
                raise

    async def send_text(
        self, chat_id: str, text: str, *, buttons: Sequence[Sequence[Button]] = ()
    ) -> None:
        await self._bot.send_message(chat_id=chat_id, text=text, reply_markup=_markup(buttons))

    async def answer_callback(self, callback: Callback, text: str | None = None) -> None:
        try:
            await self._bot.answer_callback_query(callback.query_id, text=text)
        except TelegramBadRequest as exc:
            # The query expires after a while; the user simply sees no notice.
            _log.info("callback_answer_failed", error=exc.message[:80])


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
