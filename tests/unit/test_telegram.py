"""Tests for the Telegram adapter: parsing, replies and the polling loop."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import (
    DeleteWebhook,
    EditMessageText,
    GetUpdates,
    SendMessage,
    TelegramMethod,
)
from aiogram.methods.base import TelegramType
from aiogram.types import Chat, Message, Update
from pydantic import SecretStr

from scambot.channels.telegram import TelegramAdapter, create_bot, poll_updates
from scambot.core.contracts import Command, Job
from scambot.core.security import user_hash

TOKEN = "123456:" + "A" * 35
PEPPER = SecretStr("test-pepper")

Responder = Callable[[TelegramMethod[Any]], Any]


class RecordingSession(BaseSession):
    """Answers Bot API calls from ``responder`` and records them; never touches the network."""

    def __init__(self, responder: Responder) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self._responder = responder

    # Signatures are fixed by aiogram's BaseSession, hence the ``timeout`` parameters.
    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> TelegramType:
        self.calls.append(method)
        result = self._responder(method)
        if isinstance(result, Exception):
            raise result
        return cast(TelegramType, result)

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,  # noqa: ASYNC109
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        yield b""

    async def close(self) -> None:
        return None


def sent_message(message_id: int, chat_id: int = 42) -> Message:
    return Message(
        message_id=message_id,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type="private"),
    )


def make_adapter(responder: Responder) -> tuple[TelegramAdapter, RecordingSession]:
    session = RecordingSession(responder)
    return TelegramAdapter(Bot(TOKEN, session=session), PEPPER), session


def raw_update(update_id: int = 100, /, **message: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "message_id": 5,
        "date": 1_700_000_000,
        "chat": {"id": 42, "type": "private"},
        "from": {"id": 42, "is_bot": False, "first_name": "Asha", "language_code": "hi"},
    }
    return {"update_id": update_id, "message": base | message}


@pytest.fixture
def adapter() -> TelegramAdapter:
    return make_adapter(lambda _method: None)[0]


def test_text_message_becomes_a_text_job(adapter: TelegramAdapter) -> None:
    job = adapter.parse_update(raw_update(text="Win a free iPhone"), trace_id="t1")

    assert isinstance(job, Job)
    assert adapter.channel == "telegram"
    assert job.job_id == "telegram:100"
    assert job.trace_id == "t1"
    assert job.channel == "telegram"
    assert job.chat_id == "42"
    assert job.user_hash == user_hash(42, PEPPER)
    assert job.kind == "text"
    assert job.text == "Win a free iPhone"
    assert job.media_ref is None
    assert job.preferred_lang == "hi"
    assert job.ack_message_id is None
    assert job.received_at.tzinfo is not None


def test_user_id_is_never_stored_in_clear(adapter: TelegramAdapter) -> None:
    job = adapter.parse_update(raw_update(text="hello"), trace_id="t")

    assert isinstance(job, Job)
    assert "42" not in job.user_hash


@pytest.mark.parametrize(
    ("message", "kind", "media_ref"),
    [
        (
            {
                "photo": [
                    {"file_id": "small", "file_unique_id": "s", "width": 90, "height": 90},
                    {"file_id": "large", "file_unique_id": "l", "width": 900, "height": 900},
                ],
                "caption": "Flood in Mumbai today",
            },
            "image",
            "large",
        ),
        (
            {
                "video": {
                    "file_id": "vid",
                    "file_unique_id": "v",
                    "width": 1,
                    "height": 1,
                    "duration": 3,
                },
                "caption": "Flood in Mumbai today",
            },
            "video",
            "vid",
        ),
        ({"voice": {"file_id": "vo", "file_unique_id": "o", "duration": 4}}, "voice", "vo"),
        ({"audio": {"file_id": "au", "file_unique_id": "a", "duration": 4}}, "audio", "au"),
    ],
)
def test_media_messages_become_media_jobs(
    adapter: TelegramAdapter, message: dict[str, Any], kind: str, media_ref: str
) -> None:
    job = adapter.parse_update(raw_update(**message), trace_id="t")

    assert isinstance(job, Job)
    assert job.kind == kind
    assert job.media_ref == media_ref
    assert job.text == message.get("caption")


@pytest.mark.parametrize(
    ("text", "name", "args"),
    [
        ("/start", "start", ""),
        ("/HELP", "help", ""),
        ("/privacy@ScamCheckerBot", "privacy", ""),
        ("/start campaign_42", "start", "campaign_42"),
        ("/unknown_thing", "unknown_thing", ""),
    ],
)
def test_slash_commands_become_commands(
    adapter: TelegramAdapter, text: str, name: str, args: str
) -> None:
    command = adapter.parse_update(raw_update(text=text), trace_id="t")

    assert isinstance(command, Command)
    assert command.command_id == "telegram:100"
    assert command.name == name
    assert command.args == args
    assert command.preferred_lang == "hi"


def test_text_that_merely_starts_with_a_slash_is_checked(adapter: TelegramAdapter) -> None:
    job = adapter.parse_update(raw_update(text="/ this is not a command"), trace_id="t")

    assert isinstance(job, Job)


def test_caption_that_looks_like_a_command_stays_a_media_job(adapter: TelegramAdapter) -> None:
    photo = [{"file_id": "p", "file_unique_id": "u", "width": 1, "height": 1}]

    job = adapter.parse_update(raw_update(photo=photo, caption="/start"), trace_id="t")

    assert isinstance(job, Job)
    assert (job.kind, job.text) == ("image", "/start")


def test_forwarded_command_text_is_checked_not_answered(adapter: TelegramAdapter) -> None:
    origin = {"type": "hidden_user", "date": 1_690_000_000, "sender_user_name": "Someone"}

    job = adapter.parse_update(raw_update(text="/start", forward_origin=origin), trace_id="t")

    assert isinstance(job, Job)
    assert job.text == "/start"


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (
            {
                "type": "user",
                "date": 1_690_000_000,
                "sender_user": {"id": 9, "is_bot": False, "first_name": "Ravi", "last_name": "K"},
            },
            {"type": "user", "name": "Ravi K"},
        ),
        (
            {"type": "hidden_user", "date": 1_690_000_000, "sender_user_name": "Someone"},
            {"type": "hidden_user", "name": "Someone"},
        ),
        (
            {
                "type": "chat",
                "date": 1_690_000_000,
                "sender_chat": {"id": -1, "type": "group", "title": "Family"},
            },
            {"type": "chat", "name": "Family"},
        ),
        (
            {
                "type": "channel",
                "date": 1_690_000_000,
                "message_id": 3,
                "chat": {"id": -100, "type": "channel", "title": "News"},
            },
            {"type": "channel", "name": "News"},
        ),
    ],
)
def test_forward_origin_is_kept(
    adapter: TelegramAdapter, origin: dict[str, Any], expected: dict[str, str]
) -> None:
    job = adapter.parse_update(raw_update(text="hi", forward_origin=origin), trace_id="t")

    assert isinstance(job, Job)
    assert job.forward_meta == expected | {"date": "2023-07-22T04:26:40+00:00"}


@pytest.mark.parametrize(
    "raw",
    [
        {"update_id": 1, "edited_message": raw_update(text="x")["message"]},
        raw_update(text="x", chat={"id": -5, "type": "group", "title": "G"}),
        raw_update(text="x", **{"from": {"id": 1, "is_bot": True, "first_name": "B"}}),
        raw_update(
            sticker={
                "file_id": "s",
                "file_unique_id": "u",
                "type": "regular",
                "width": 1,
                "height": 1,
                "is_animated": False,
                "is_video": False,
            }
        ),
        raw_update(document={"file_id": "d", "file_unique_id": "u"}),
        {
            "update_id": 1,
            "callback_query": {
                "id": "1",
                "from": {"id": 42, "is_bot": False, "first_name": "Asha"},
                "chat_instance": "c",
            },
        },
        {
            "update_id": 1,
            "channel_post": raw_update(text="x", chat={"id": -9, "type": "channel", "title": "C"})[
                "message"
            ],
        },
        {"update_id": 1, "message": "not-an-object"},
        {"update_id": "not-a-number"},
        {},
    ],
    ids=[
        "edited",
        "group",
        "from-bot",
        "sticker",
        "document",
        "callback-query",
        "channel-post",
        "message-not-object",
        "invalid",
        "empty",
    ],
)
def test_other_updates_are_ignored(adapter: TelegramAdapter, raw: dict[str, Any]) -> None:
    assert adapter.parse_update(raw, trace_id="t") is None


def a_job(ack_message_id: str | None = None) -> Job:
    return Job(
        job_id="telegram:1",
        trace_id="t",
        channel="telegram",
        chat_id="42",
        user_hash="h",
        kind="text",
        text="x",
        ack_message_id=ack_message_id,
        received_at=datetime.now(UTC),
    )


async def test_send_ack_returns_the_message_id() -> None:
    adapter, session = make_adapter(lambda _method: sent_message(77))

    assert await adapter.send_ack(a_job(), "Checking…") == "77"
    (call,) = session.calls
    assert isinstance(call, SendMessage)
    assert (call.chat_id, call.text) == ("42", "Checking…")


async def test_send_verdict_edits_the_ack() -> None:
    adapter, session = make_adapter(lambda _method: True)

    await adapter.send_verdict(a_job(ack_message_id="77"), "Verdict")

    (call,) = session.calls
    assert isinstance(call, EditMessageText)
    assert (call.chat_id, call.message_id, call.text) == ("42", 77, "Verdict")


async def test_send_verdict_without_ack_sends_a_new_message() -> None:
    adapter, session = make_adapter(lambda _method: sent_message(78))

    await adapter.send_verdict(a_job(), "Verdict")

    (call,) = session.calls
    assert isinstance(call, SendMessage)


async def test_repeated_edit_with_same_text_is_not_an_error() -> None:
    def not_modified(method: TelegramMethod[Any]) -> Exception:
        return TelegramBadRequest(method, "Bad Request: message is not modified: same content")

    adapter, _ = make_adapter(not_modified)

    await adapter.send_verdict(a_job(ack_message_id="77"), "Verdict")


async def test_other_edit_errors_propagate() -> None:
    def gone(method: TelegramMethod[Any]) -> Exception:
        return TelegramBadRequest(method, "Bad Request: message to edit not found")

    adapter, _ = make_adapter(gone)

    with pytest.raises(TelegramBadRequest):
        await adapter.send_verdict(a_job(ack_message_id="77"), "Verdict")


async def test_send_text() -> None:
    adapter, session = make_adapter(lambda _method: sent_message(79))

    await adapter.send_text("42", "Hello")

    (call,) = session.calls
    assert isinstance(call, SendMessage)
    assert call.text == "Hello"


async def test_poll_updates_feeds_handler_and_survives_errors() -> None:
    stop = asyncio.Event()
    batches: list[Any] = [
        TelegramNetworkError(GetUpdates(), "timeout"),
        [
            Update.model_validate(raw_update(1, text="one")),
            # aiogram fills the unset link-preview fields with placeholders; they must not
            # break the conversion back to webhook JSON.
            Update.model_validate(
                raw_update(
                    2, text="two https://example.com", link_preview_options={"is_disabled": True}
                )
            ),
        ],
    ]
    offsets: list[int | None] = []

    def responder(method: TelegramMethod[Any]) -> Any:
        if isinstance(method, DeleteWebhook):
            return True
        assert isinstance(method, GetUpdates)
        offsets.append(method.offset)
        if batches:
            return batches.pop(0)
        stop.set()
        return []

    session = RecordingSession(responder)
    received: list[dict[str, Any]] = []

    async def handler(update: dict[str, Any]) -> None:
        received.append(update)
        if update["update_id"] == 1:
            msg = "handler failure must not stop polling"
            raise RuntimeError(msg)

    await asyncio.wait_for(
        poll_updates(Bot(TOKEN, session=session), handler, stop=stop, backoff_start=0), 5
    )

    assert isinstance(session.calls[0], DeleteWebhook)
    assert [u["update_id"] for u in received] == [1, 2]
    assert received[1]["message"]["from"]["id"] == 42
    assert received[1]["message"]["link_preview_options"] == {"is_disabled": True}
    assert offsets == [None, None, 3]


async def test_create_bot_uses_the_configured_api_url() -> None:
    bot = create_bot(SecretStr(TOKEN), "http://127.0.0.1:8081/")
    try:
        assert bot.token == TOKEN
        assert bot.session.api.api_url(TOKEN, "getMe") == f"http://127.0.0.1:8081/bot{TOKEN}/getMe"
    finally:
        await bot.session.close()
