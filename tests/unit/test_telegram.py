"""Tests for the Telegram adapter: parsing, replies and the polling loop."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiohttp
import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import (
    AnswerCallbackQuery,
    DeleteWebhook,
    EditMessageText,
    GetFile,
    GetUpdates,
    SendMessage,
    TelegramMethod,
)
from aiogram.methods.base import TelegramType
from aiogram.types import Chat, File, Message, Update
from pydantic import SecretStr

from scambot.channels.base import Button
from scambot.channels.telegram import ALLOWED_UPDATES, TelegramAdapter, create_bot, poll_updates
from scambot.core.contracts import Callback, Command, Job, Rejection
from scambot.core.media import MediaRejected, MediaUnavailable
from scambot.core.media_store import LocalMediaStore
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


def photo_of(size: int | None = None) -> dict[str, Any]:
    return {"file_id": "p", "file_unique_id": "u", "width": 9, "height": 9, "file_size": size}


@pytest.mark.parametrize(
    ("message", "kind", "source"),
    [
        ({"photo": [photo_of(100)]}, "image", "photo"),
        (
            {"document": {"file_id": "d", "file_unique_id": "u", "mime_type": "image/jpeg"}},
            "image",
            "document",
        ),
        (
            {"video_note": {"file_id": "n", "file_unique_id": "u", "length": 240, "duration": 5}},
            "video",
            "video_note",
        ),
        (
            {
                "animation": {
                    "file_id": "g",
                    "file_unique_id": "u",
                    "width": 1,
                    "height": 1,
                    "duration": 2,
                    "mime_type": "video/mp4",
                },
                "document": {"file_id": "g", "file_unique_id": "u", "mime_type": "video/mp4"},
            },
            "video",
            "animation",
        ),
        (
            {
                "voice": {
                    "file_id": "v",
                    "file_unique_id": "u",
                    "duration": 4,
                    "mime_type": "audio/ogg",
                }
            },
            "voice",
            "voice",
        ),
        (
            {
                "video": {
                    "file_id": "v",
                    "file_unique_id": "u",
                    "width": 1,
                    "height": 1,
                    "duration": 4,
                    "file_size": 99,
                }
            },
            "video",
            "video",
        ),
        ({"audio": {"file_id": "a", "file_unique_id": "u", "duration": 4}}, "audio", "audio"),
    ],
)
def test_media_metadata_is_declared(
    adapter: TelegramAdapter, message: dict[str, Any], kind: str, source: str
) -> None:
    job = adapter.parse_update(raw_update(**message), trace_id="t")

    assert isinstance(job, Job)
    assert job.kind == kind
    assert job.media is not None
    assert job.media.source == source


def test_declared_size_and_type_are_kept(adapter: TelegramAdapter) -> None:
    voice = {"file_id": "v", "file_unique_id": "u", "duration": 4, "file_size": 25_000_000}
    job = adapter.parse_update(raw_update(voice=voice), trace_id="t")

    assert isinstance(job, Job)
    assert job.media is not None
    assert (job.media.file_size, job.media.duration_s) == (25_000_000, 4)


@pytest.mark.parametrize("mime", ["application/x-msdownload", None])
def test_unsupported_document_is_a_rejection(adapter: TelegramAdapter, mime: str | None) -> None:
    document = {"file_id": "d", "file_unique_id": "u", "mime_type": mime}

    result = adapter.parse_update(raw_update(document=document), trace_id="t")

    assert isinstance(result, Rejection)
    assert result.reason == "unsupported_type"
    assert result.rejection_id == "telegram:100"
    assert result.preferred_lang == "hi"


class StreamSession(RecordingSession):
    """A session that answers file downloads from memory."""

    def __init__(self, responder: Responder, chunks: list[bytes], error: Exception | None = None):
        super().__init__(responder)
        self.chunks = chunks
        self.error = error
        self.urls: list[str] = []

    async def stream_content(
        self, url: str, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[bytes, None]:
        self.urls.append(url)
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


def downloader(
    chunks: list[bytes],
    *,
    error: Exception | None = None,
    file: File | Exception | None = None,
) -> tuple[TelegramAdapter, StreamSession]:
    found = file or File(file_id="f", file_unique_id="u", file_size=10, file_path="photos/a.jpg")

    def respond(method: TelegramMethod[Any]) -> Any:
        assert isinstance(method, GetFile)
        return found

    session = StreamSession(respond, chunks, error)
    return TelegramAdapter(Bot(TOKEN, session=session), PEPPER), session


def media_job() -> Job:
    return Job(
        job_id="telegram:1",
        trace_id="t",
        channel="telegram",
        chat_id="42",
        user_hash="h",
        kind="image",
        media_ref="f",
        received_at=datetime.now(UTC),
    )


async def test_download_streams_into_the_store(tmp_path: Path) -> None:
    store = LocalMediaStore(tmp_path / "m")
    adapter, session = downloader([b"abc", b"def"])

    path = await adapter.download_media(media_job(), store, "k", max_bytes=100, timeout_s=5)

    assert path.read_bytes() == b"abcdef"
    assert session.urls == [f"https://api.telegram.org/file/bot{TOKEN}/photos/a.jpg"]


async def test_download_over_the_cap_leaves_no_file(tmp_path: Path) -> None:
    store = LocalMediaStore(tmp_path / "m")
    adapter, _ = downloader([b"a" * 60, b"b" * 60])

    with pytest.raises(MediaRejected) as raised:
        await adapter.download_media(media_job(), store, "k", max_bytes=100, timeout_s=5)

    assert raised.value.reason == "too_large"
    assert list((tmp_path / "m" / "k").iterdir()) == []


async def test_declared_file_size_over_the_cap_is_refused_before_downloading(
    tmp_path: Path,
) -> None:
    file = File(file_id="f", file_unique_id="u", file_size=500, file_path="a")
    adapter, session = downloader([b"x"], file=file)

    with pytest.raises(MediaRejected) as raised:
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=100, timeout_s=5
        )

    assert raised.value.reason == "too_large"
    assert session.urls == []


@pytest.mark.parametrize(
    ("message", "reason"),
    [("Bad Request: file is too big", "too_large"), ("Bad Request: wrong file_id", "unreadable")],
)
async def test_get_file_bad_request(tmp_path: Path, message: str, reason: str) -> None:
    error = TelegramBadRequest(method=GetFile(file_id="f"), message=message)
    adapter, _ = downloader([], file=error)

    with pytest.raises(MediaRejected) as raised:
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=100, timeout_s=5
        )

    assert raised.value.reason == reason


async def test_file_without_a_path_is_unreadable(tmp_path: Path) -> None:
    adapter, _ = downloader([], file=File(file_id="f", file_unique_id="u"))

    with pytest.raises(MediaRejected) as raised:
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=100, timeout_s=5
        )

    assert raised.value.reason == "unreadable"


async def test_network_failure_is_transient_and_does_not_leak_the_token(tmp_path: Path) -> None:
    adapter, _ = downloader([b"a"], error=aiohttp.ClientConnectionError(f"failed bot{TOKEN}"))

    with pytest.raises(MediaUnavailable) as raised:
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=100, timeout_s=5
        )

    assert "AAAA" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


async def test_client_error_status_is_unreadable_and_server_error_transient(
    tmp_path: Path,
) -> None:
    def failure(status: int) -> aiohttp.ClientResponseError:
        info = SimpleNamespace(real_url="https://example.invalid/file")
        return aiohttp.ClientResponseError(cast(Any, info), (), status=status)

    adapter, _ = downloader([], error=failure(404))
    with pytest.raises(MediaRejected):
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=100, timeout_s=5
        )

    adapter, _ = downloader([], error=failure(502))
    with pytest.raises(MediaUnavailable):
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=100, timeout_s=5
        )


async def test_download_without_a_reference_is_unreadable(tmp_path: Path) -> None:
    adapter, _ = downloader([])
    job = media_job().model_copy(update={"media_ref": None})

    with pytest.raises(MediaRejected):
        await adapter.download_media(job, LocalMediaStore(tmp_path), "k", max_bytes=1, timeout_s=1)


async def test_get_file_network_error_is_transient(tmp_path: Path) -> None:
    error = TelegramNetworkError(method=GetFile(file_id="f"), message="down")
    adapter, _ = downloader([], file=error)

    with pytest.raises(MediaUnavailable):
        await adapter.download_media(
            media_job(), LocalMediaStore(tmp_path), "k", max_bytes=1, timeout_s=1
        )


def callback_update(
    data: str, *, chat_type: str = "private", is_bot: bool = False
) -> dict[str, Any]:
    return {
        "update_id": 300,
        "callback_query": {
            "id": "cbq1",
            "chat_instance": "ci",
            "from": {"id": 42, "is_bot": is_bot, "first_name": "Asha", "language_code": "hi"},
            "data": data,
            "message": {
                "message_id": 5,
                "date": 1_700_000_000,
                "chat": {"id": 42, "type": chat_type},
            },
        },
    }


REQUEST_ID = "123e4567-e89b-12d3-a456-426614174000"


@pytest.mark.parametrize(
    ("data", "action", "value"),
    [
        (f"d:{REQUEST_ID}", "details", REQUEST_ID),
        (f"h:{REQUEST_ID}", "helpful", REQUEST_ID),
        (f"w:{REQUEST_ID}", "wrong", REQUEST_ID),
        ("l:hi-Latn", "language", "hi-Latn"),
    ],
)
def test_button_presses_are_parsed(
    adapter: TelegramAdapter, data: str, action: str, value: str
) -> None:
    parsed = adapter.parse_update(callback_update(data), trace_id="t")

    assert isinstance(parsed, Callback)
    assert (parsed.action, parsed.value) == (action, value)
    assert parsed.callback_id == "telegram:cb:cbq1"
    assert parsed.chat_id == "42"
    assert parsed.preferred_lang == "hi"
    assert parsed.user_hash == user_hash(42, PEPPER)


@pytest.mark.parametrize(
    "data", ["", "x:abc", "d:hi", f"l:{REQUEST_ID}", "d:" + "z" * 36, f"d:{REQUEST_ID}!"]
)
def test_bad_button_data_is_ignored(adapter: TelegramAdapter, data: str) -> None:
    assert adapter.parse_update(callback_update(data), trace_id="t") is None


def test_button_presses_from_groups_or_bots_are_ignored(adapter: TelegramAdapter) -> None:
    group = callback_update(f"d:{REQUEST_ID}", chat_type="group")
    bot = callback_update(f"d:{REQUEST_ID}", is_bot=True)

    assert adapter.parse_update(group, trace_id="t") is None
    assert adapter.parse_update(bot, trace_id="t") is None


def test_a_press_on_an_inaccessible_message_still_gets_an_answer(adapter: TelegramAdapter) -> None:
    raw = callback_update(f"d:{REQUEST_ID}")
    raw["callback_query"]["message"] = {
        "message_id": 5,
        "date": 0,
        "chat": {"id": 42, "type": "private"},
    }

    parsed = adapter.parse_update(raw, trace_id="t")

    assert isinstance(parsed, Callback)
    assert (parsed.action, parsed.chat_id, parsed.preferred_lang) == ("expired", "42", "hi")


def test_callbacks_are_requested_from_telegram() -> None:
    assert "callback_query" in ALLOWED_UPDATES


async def test_verdict_buttons_are_sent_as_one_row() -> None:
    adapter, session = make_adapter(lambda _m: True)

    await adapter.send_verdict(
        a_job(ack_message_id="77"),
        "Verdict",
        buttons=[Button("Details", "d:1"), Button("Hi", "h:1")],
    )

    (call,) = session.calls
    assert isinstance(call, EditMessageText)
    rows = call.reply_markup.inline_keyboard  # type: ignore[union-attr]
    assert [[b.callback_data for b in row] for row in rows] == [["d:1", "h:1"]]


async def test_text_keyboards_and_answering_a_press() -> None:
    adapter, session = make_adapter(lambda _m: sent_message(1))

    await adapter.send_text("42", "Choose", buttons=[[Button("A", "l:en")], [Button("B", "l:hi")]])
    await adapter.send_text("42", "Plain")
    await adapter.answer_callback(
        Callback(
            callback_id="c",
            query_id="q",
            trace_id="t",
            channel="telegram",
            chat_id="42",
            user_hash="u",
            action="helpful",
            value=REQUEST_ID,
        ),
        "Thanks",
    )

    keyboard, plain, answer = session.calls
    assert len(keyboard.reply_markup.inline_keyboard) == 2  # type: ignore[attr-defined]
    assert plain.reply_markup is None  # type: ignore[attr-defined]
    assert isinstance(answer, AnswerCallbackQuery)
    assert answer.text == "Thanks"


async def test_an_expired_press_is_not_an_error() -> None:
    adapter, _ = make_adapter(lambda m: TelegramBadRequest(method=m, message="query is too old"))
    callback = Callback(
        callback_id="c",
        query_id="q",
        trace_id="t",
        channel="telegram",
        chat_id="42",
        user_hash="u",
        action="details",
        value=REQUEST_ID,
    )

    await adapter.answer_callback(callback)
