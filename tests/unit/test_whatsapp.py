"""WhatsApp adapter: signatures, parsing, media download and sends, with the Graph API mocked."""

import hashlib
import hmac
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from scambot.channels.base import Button
from scambot.channels.whatsapp import (
    WhatsAppAdapter,
    WhatsAppError,
    normalise_number,
    verify_signature,
)
from scambot.core.contracts import Callback, Command, Job, Rejection
from scambot.core.locales import Locales
from scambot.core.logging import configure_logging
from scambot.core.media import MediaRejected, MediaUnavailable
from scambot.core.media_store import LocalMediaStore
from scambot.core.security import user_hash
from scambot.core.settings import DEV_USER_ID_PEPPER
from tests.whatsapp import (
    ACCESS_TOKEN,
    APP_SECRET,
    CDN,
    GRAPH,
    PHONE_ID,
    SENDER,
    STRANGER,
    button,
    message,
    raw,
    text,
)

MESSAGES = f"{GRAPH}/{PHONE_ID}/messages"
REQUEST_ID = "0b1c2d3e-0000-4000-8000-000000000001"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(follow_redirects=False) as client:
        yield client


def make(client: httpx.AsyncClient, locales: Locales, **kwargs: Any) -> WhatsAppAdapter:
    options: dict[str, Any] = {
        "token": SecretStr(ACCESS_TOKEN),
        "phone_number_id": PHONE_ID,
        "api_version": "v23.0",
        "graph_url": "https://graph.facebook.com",
        "allowlist": ["+91 98000 00001"],
        "user_id_pepper": SecretStr(DEV_USER_ID_PEPPER),
        "locales": locales,
    }
    return WhatsAppAdapter(client, **{**options, **kwargs})


@pytest.fixture
def adapter(client: httpx.AsyncClient, locales: Locales) -> WhatsAppAdapter:
    return make(client, locales)


def parse(adapter: WhatsAppAdapter, msg: dict[str, Any]) -> Any:
    return adapter.parse_update(raw(msg), trace_id="t")


def job_for(adapter: WhatsAppAdapter, **kwargs: Any) -> Job:
    result = parse(adapter, text("a long enough message to be checked now", **kwargs))
    assert isinstance(result, Job)
    return result


# --------------------------------------------------------------------- signature


def test_signature_accepts_the_correct_digest() -> None:
    body = b'{"a": 1}'
    digest = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()

    assert verify_signature(body, f"sha256={digest}", SecretStr(APP_SECRET))


@pytest.mark.parametrize(
    "header", [None, "", "sha256=", "md5=abc", "sha256=" + "0" * 64, "sha256=zz" + "0" * 62]
)
def test_signature_rejects_bad_headers(header: str | None) -> None:
    assert not verify_signature(b"{}", header, SecretStr(APP_SECRET))


def test_signature_rejects_tampered_body_wrong_secret_and_empty_secret() -> None:
    good = "sha256=" + hmac.new(APP_SECRET.encode(), b"{}", hashlib.sha256).hexdigest()

    assert not verify_signature(b"{ }", good, SecretStr(APP_SECRET))
    assert not verify_signature(b"{}", good, SecretStr("other"))
    assert not verify_signature(b"{}", good, SecretStr(""))


# ------------------------------------------------------------------------ parsing


def test_text_becomes_a_job(adapter: WhatsAppAdapter) -> None:
    job = job_for(adapter)

    assert job.job_id == "whatsapp:wamid.A1"
    assert (job.kind, job.channel, job.chat_id) == ("text", "whatsapp", SENDER)
    assert job.text == "a long enough message to be checked now"
    assert job.user_hash == user_hash(SENDER, SecretStr(DEV_USER_ID_PEPPER))
    assert job.forward_meta == {}
    assert job.preferred_lang is None
    assert job.received_at.year >= 2025


def test_allowlist_is_normalised_and_others_are_ignored(
    adapter: WhatsAppAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    assert normalise_number("+91 98000 00001") == SENDER
    assert parse(adapter, text("hello there friend ok", sender=STRANGER)) is None
    assert STRANGER not in capsys.readouterr().out


def test_empty_allowlist_serves_nobody(client: httpx.AsyncClient, locales: Locales) -> None:
    adapter = make(client, locales, allowlist=[])

    assert parse(adapter, text("hello there friend ok")) is None


def test_rejected_sender_is_logged_by_hash_prefix_only(
    adapter: WhatsAppAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    parse(adapter, text("hello there friend ok", sender=STRANGER))

    out = capsys.readouterr().out
    assert "whatsapp_sender_not_allowed" in out
    assert user_hash(STRANGER, SecretStr(DEV_USER_ID_PEPPER))[:12] in out
    assert STRANGER not in out


@pytest.mark.parametrize(
    ("body", "name", "args"),
    [
        ("help", "help", ""),
        ("Help!", "help", ""),
        ("/history", "history", ""),
        ("language hindi", "language", "hindi"),
        ("DELETE", "delete", ""),
        ("Menu", "help", ""),
        ("privacy", "privacy", ""),
        ("hi", "start", ""),
        ("Namaste", "start", ""),
        ("/language", "language", ""),
    ],
)
def test_plain_word_commands(adapter: WhatsAppAdapter, body: str, name: str, args: str) -> None:
    command = parse(adapter, text(body))

    assert isinstance(command, Command)
    assert (command.name, command.args) == (name, args)
    assert command.command_id == "whatsapp:wamid.A1"


def test_a_sentence_is_not_a_command(adapter: WhatsAppAdapter) -> None:
    assert isinstance(parse(adapter, text("delete my account please")), Job)


def test_forwarded_help_is_content_to_check(adapter: WhatsAppAdapter) -> None:
    msg = text("help", context={"forwarded": True, "frequently_forwarded": True})

    job = parse(adapter, msg)

    assert isinstance(job, Job)
    assert job.forward_meta == {"type": "forwarded", "frequently_forwarded": True}


@pytest.mark.parametrize("body", ["thanks a lot", "capital of France", "ok"])
def test_small_talk_gets_the_help_text(adapter: WhatsAppAdapter, body: str) -> None:
    command = parse(adapter, text(body))

    assert isinstance(command, Command)
    assert command.name == "help"


@pytest.mark.parametrize("body", ["see bit.ly/abc", "call 9800000001", "visit https://x.example"])
def test_short_text_with_a_link_or_number_is_checked(adapter: WhatsAppAdapter, body: str) -> None:
    assert isinstance(parse(adapter, text(body)), Job)


def test_empty_text_is_ignored(adapter: WhatsAppAdapter) -> None:
    assert parse(adapter, text("   ")) is None


def test_image_with_caption(adapter: WhatsAppAdapter) -> None:
    msg = message("image", {"id": "555", "mime_type": "image/jpeg", "caption": "flood!"})

    job = parse(adapter, msg)

    assert isinstance(job, Job)
    assert (job.kind, job.media_ref, job.text) == ("image", "555", "flood!")
    assert job.media is not None
    assert (job.media.source, job.media.mime_type) == ("photo", "image/jpeg")


def test_video_audio_voice_and_document(adapter: WhatsAppAdapter) -> None:
    video = parse(adapter, message("video", {"id": "1", "mime_type": "video/mp4"}))
    audio = parse(adapter, message("audio", {"id": "2", "mime_type": "audio/mpeg"}))
    voice = parse(adapter, message("audio", {"id": "3", "mime_type": "audio/ogg", "voice": True}))
    pdf = parse(adapter, message("document", {"id": "4", "mime_type": "application/pdf"}))
    photo = parse(adapter, message("document", {"id": "5", "mime_type": "image/png"}))

    assert [(j.kind, j.media.source) for j in (video, audio, voice, photo) if j.media] == [
        ("video", "video"),
        ("audio", "audio"),
        ("voice", "voice"),
        ("image", "document"),
    ]
    assert isinstance(pdf, Rejection)
    assert pdf.reason == "unsupported_type"


@pytest.mark.parametrize("kind", ["sticker", "location", "contacts", "order", "button"])
def test_unsupported_types_are_rejected(adapter: WhatsAppAdapter, kind: str) -> None:
    result = parse(adapter, message(kind, {}))

    assert isinstance(result, Rejection)
    assert result.reason == "unsupported_type"


@pytest.mark.parametrize("kind", ["reaction", "system", "unsupported", "request_welcome", "x"])
def test_ignored_types(adapter: WhatsAppAdapter, kind: str) -> None:
    result = parse(adapter, message(kind, {}))

    assert result is None or kind == "x"


def test_malformed_shapes_are_ignored(adapter: WhatsAppAdapter) -> None:
    assert adapter.parse_update({}, trace_id="t") is None
    assert adapter.parse_update({"message": {"id": 1}}, trace_id="t") is None
    assert (
        adapter.parse_update({"message": {"id": "a", "from": "", "type": "text"}}, trace_id="t")
        is None
    )


def test_stranger_media_is_not_rejected_either(adapter: WhatsAppAdapter) -> None:
    assert parse(adapter, message("sticker", {}, sender=STRANGER)) is None


@pytest.mark.parametrize(
    ("data", "action", "value"),
    [
        (f"d:{REQUEST_ID}", "details", REQUEST_ID),
        (f"h:{REQUEST_ID}", "helpful", REQUEST_ID),
        (f"w:{REQUEST_ID}", "wrong", REQUEST_ID),
        ("l:hi-Latn", "language", "hi-Latn"),
    ],
)
def test_button_replies(adapter: WhatsAppAdapter, data: str, action: str, value: str) -> None:
    callback = parse(adapter, button(data))

    assert isinstance(callback, Callback)
    assert (callback.action, callback.value) == (action, value)
    assert callback.callback_id == "whatsapp:cb:wamid.A1"
    assert callback.query_id == "wamid.A1"


def test_garbage_button_ids_and_list_replies_are_ignored(adapter: WhatsAppAdapter) -> None:
    assert parse(adapter, button("zzz")) is None
    assert parse(adapter, button(f"l:{REQUEST_ID}")) is None
    listed = message("interactive", None, interactive={"type": "list_reply", "list_reply": {}})
    assert parse(adapter, listed) is None


# ---------------------------------------------------------------------- download


@pytest.fixture
def media_store(tmp_path: Path) -> LocalMediaStore:
    return LocalMediaStore(tmp_path)


def media_job(adapter: WhatsAppAdapter, ref: str = "555", mime: str = "image/jpeg") -> Job:
    job = parse(adapter, message("image", {"id": ref, "mime_type": mime}))
    assert isinstance(job, Job)
    return job


def lookup(mock: respx.MockRouter, **info: Any) -> respx.Route:
    body = {"url": CDN, "mime_type": "image/jpeg", "file_size": 5, "id": "555"} | info
    return mock.get(f"{GRAPH}/555").respond(200, json=body)


@respx.mock
async def test_download_uses_the_token_on_both_calls(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore
) -> None:
    info = lookup(respx.mock)
    cdn = respx.mock.get(CDN).respond(200, content=b"hello")

    path = await adapter.download_media(
        media_job(adapter), media_store, "k", max_bytes=100, timeout_s=5
    )

    assert path.read_bytes() == b"hello"
    for route in (info, cdn):
        assert route.calls.last.request.headers["authorization"] == f"Bearer {ACCESS_TOKEN}"


@respx.mock
@pytest.mark.parametrize(
    ("status", "error"),
    [(404, MediaRejected), (400, MediaRejected), (500, MediaUnavailable), (429, MediaUnavailable)],
)
async def test_lookup_errors(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore, status: int, error: type[Exception]
) -> None:
    respx.mock.get(f"{GRAPH}/555").respond(status)

    with pytest.raises(error):
        await adapter.download_media(media_job(adapter), media_store, "k", max_bytes=9, timeout_s=5)


@respx.mock
async def test_network_errors_do_not_leak_the_token_or_url(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore
) -> None:
    respx.mock.get(f"{GRAPH}/555").mock(side_effect=httpx.ConnectError(f"boom {ACCESS_TOKEN}"))

    with pytest.raises(MediaUnavailable) as caught:
        await adapter.download_media(media_job(adapter), media_store, "k", max_bytes=9, timeout_s=5)

    assert caught.value.__cause__ is None
    assert ACCESS_TOKEN not in str(caught.value) + repr(caught.value)
    assert "graph.facebook" not in str(caught.value)


@respx.mock
async def test_declared_size_wrong_mime_and_stream_overflow(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore, tmp_path: Path
) -> None:
    job = media_job(adapter)
    lookup(respx.mock, file_size=500)
    with pytest.raises(MediaRejected) as big:
        await adapter.download_media(job, media_store, "k", max_bytes=100, timeout_s=5)
    assert big.value.reason == "too_large"

    lookup(respx.mock, mime_type="application/pdf")
    with pytest.raises(MediaRejected) as wrong:
        await adapter.download_media(job, media_store, "k", max_bytes=100, timeout_s=5)
    assert wrong.value.reason == "unsupported_type"

    lookup(respx.mock, file_size=None)
    respx.mock.get(CDN).respond(200, content=b"x" * 300)
    with pytest.raises(MediaRejected) as over:
        await adapter.download_media(job, media_store, "k", max_bytes=100, timeout_s=5)
    assert over.value.reason == "too_large"
    assert not _parts(tmp_path)


@respx.mock
@pytest.mark.parametrize(
    "url",
    [
        "http://lookaside.fbsbx.com/x",
        "https://evil.example/x",
        "https://fbsbx.com.evil.example/x",
        "https://user:pw@lookaside.fbsbx.com/x",
    ],
)
async def test_untrusted_media_urls_are_never_requested(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore, url: str
) -> None:
    lookup(respx.mock, url=url)
    other = respx.mock.route().respond(200)

    with pytest.raises(MediaRejected) as caught:
        await adapter.download_media(
            media_job(adapter), media_store, "k", max_bytes=100, timeout_s=5
        )

    assert caught.value.reason == "unreadable"
    assert not other.called


async def test_media_id_cannot_change_the_path(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore
) -> None:
    with pytest.raises(MediaRejected):
        await adapter.download_media(
            media_job(adapter, ref="../x"), media_store, "k", max_bytes=100, timeout_s=5
        )


@respx.mock
async def test_cdn_errors(adapter: WhatsAppAdapter, media_store: LocalMediaStore) -> None:
    lookup(respx.mock)
    cdn = respx.mock.get(CDN)
    cdn.respond(403)
    with pytest.raises(MediaRejected):
        await adapter.download_media(media_job(adapter), media_store, "k", max_bytes=9, timeout_s=5)
    cdn.respond(503)
    with pytest.raises(MediaUnavailable):
        await adapter.download_media(media_job(adapter), media_store, "k", max_bytes=9, timeout_s=5)
    cdn.mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(MediaUnavailable) as caught:
        await adapter.download_media(media_job(adapter), media_store, "k", max_bytes=9, timeout_s=5)
    assert caught.value.__cause__ is None


def _parts(root: Path) -> list[Path]:
    return list(root.rglob("*.part"))


# ------------------------------------------------------------------------- sends


def sent(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


def ok(wamid: str = "wamid.OUT") -> httpx.Response:
    return httpx.Response(200, json={"messages": [{"id": wamid}]})


BUTTONS = [
    Button("Details", f"d:{REQUEST_ID}"),
    Button("👍 Helpful", f"h:{REQUEST_ID}"),
    Button("👎 Wrong", f"w:{REQUEST_ID}"),
]


@respx.mock
async def test_ack_marks_as_read_then_quotes_the_message(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(side_effect=[ok("read"), ok("wamid.ACK")])

    ack = await adapter.send_ack(job_for(adapter), "Checking…")

    assert ack == "wamid.ACK"
    read, message_out = sent(route)
    assert read["status"] == "read"
    assert read["message_id"] == "wamid.A1"
    assert read["typing_indicator"] == {"type": "text"}
    assert message_out["to"] == SENDER
    assert message_out["context"] == {"message_id": "wamid.A1"}
    assert message_out["text"]["body"] == "Checking…"


@respx.mock
async def test_mark_read_failure_is_swallowed(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(side_effect=[httpx.ConnectError("x"), ok("wamid.ACK")])

    assert await adapter.send_ack(job_for(adapter), "Checking…") == "wamid.ACK"
    assert len(route.calls) == 2


@respx.mock
async def test_mark_read_can_be_disabled(client: httpx.AsyncClient, locales: Locales) -> None:
    adapter = make(client, locales, mark_read=False)
    route = respx.mock.post(MESSAGES).mock(return_value=ok())

    await adapter.send_ack(job_for(adapter), "Checking…")

    assert [p.get("type") for p in sent(route)] == ["text"]


@respx.mock
async def test_short_verdict_is_one_interactive_message(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(return_value=ok())

    await adapter.send_verdict(job_for(adapter), "verdict /help", buttons=BUTTONS)

    (payload,) = sent(route)
    assert payload["type"] == "interactive"
    assert payload["interactive"]["body"]["text"] == "verdict help"
    rows = payload["interactive"]["action"]["buttons"]
    assert [r["reply"]["id"] for r in rows] == [b.data for b in BUTTONS]
    assert [r["reply"]["title"] for r in rows] == ["Details", "👍 Helpful", "👎 Wrong"]
    assert payload["context"] == {"message_id": "wamid.A1"}


@respx.mock
async def test_long_verdict_is_text_then_a_separate_prompt(
    adapter: WhatsAppAdapter, locales: Locales
) -> None:
    route = respx.mock.post(MESSAGES).mock(return_value=ok())

    await adapter.send_verdict(job_for(adapter), "x" * 1500, buttons=BUTTONS)

    first, second = sent(route)
    assert first["type"] == "text"
    assert second["interactive"]["body"]["text"] == locales.get("en").buttons.prompt


@respx.mock
async def test_very_long_text_is_split_on_paragraphs(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(return_value=ok())
    body = "\n\n".join(["p" * 3000, "q" * 3000, "r" * 5000])

    await adapter.send_text(SENDER, body)

    parts = [p["text"]["body"] for p in sent(route)]
    assert [len(p) for p in parts] == [3000, 3000, 4096, 904]
    assert all(len(p) <= 4096 for p in parts)


@respx.mock
async def test_button_titles_are_truncated_and_limited_to_three(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(return_value=ok())
    rows = [
        [Button("a" * 30, "l:en"), Button("b", "l:hi")],
        [Button("c", "l:x1"), Button("d", "l:x2")],
    ]

    await adapter.send_text(SENDER, "Choose", buttons=rows)

    (payload,) = sent(route)
    buttons = payload["interactive"]["action"]["buttons"]
    assert len(buttons) == 3
    assert buttons[0]["reply"]["title"] == "a" * 19 + "…"


@respx.mock
async def test_slash_commands_become_plain_words(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(return_value=ok())

    await adapter.send_text(SENDER, "Send /delete or /language; see https://x.example/help")

    assert sent(route)[0]["text"]["body"] == "Send delete or language; see https://x.example/help"


@respx.mock
@pytest.mark.parametrize("code", [131047, 131030, 131026])
async def test_window_and_recipient_errors_are_dropped(adapter: WhatsAppAdapter, code: int) -> None:
    respx.mock.post(MESSAGES).respond(
        400, json={"error": {"code": code, "type": "x", "message": "secret"}}
    )

    await adapter.send_text(SENDER, "hi")
    assert await adapter.send_ack(job_for(adapter), "Checking…") == "dropped"


@respx.mock
@pytest.mark.parametrize("status", [429, 500, 400])
async def test_other_errors_raise(adapter: WhatsAppAdapter, status: int) -> None:
    respx.mock.post(MESSAGES).respond(status, json={"error": {"code": 100, "message": "echo"}})

    with pytest.raises(WhatsAppError) as caught:
        await adapter.send_text(SENDER, "hi")

    assert caught.value.code == 100
    assert "echo" not in str(caught.value)


@respx.mock
async def test_network_error_hides_the_token(adapter: WhatsAppAdapter) -> None:
    respx.mock.post(MESSAGES).mock(side_effect=httpx.ConnectError(ACCESS_TOKEN))

    with pytest.raises(WhatsAppError) as caught:
        await adapter.send_text(SENDER, "hi")

    assert caught.value.code is None
    assert caught.value.__cause__ is None
    assert ACCESS_TOKEN not in str(caught.value)


@respx.mock
async def test_answer_callback_sends_the_notice_or_nothing(adapter: WhatsAppAdapter) -> None:
    route = respx.mock.post(MESSAGES).mock(return_value=ok())
    callback = parse(adapter, button(f"h:{REQUEST_ID}"))
    assert isinstance(callback, Callback)

    await adapter.answer_callback(callback, "Thanks")
    await adapter.answer_callback(callback, None)

    assert [p["text"]["body"] for p in sent(route)] == ["Thanks"]
    assert re.fullmatch(r"\d+", sent(route)[0]["to"])


@respx.mock
async def test_a_redirect_from_the_cdn_is_rejected_not_followed(
    adapter: WhatsAppAdapter, media_store: LocalMediaStore
) -> None:
    lookup(respx.mock)
    respx.mock.get(CDN).respond(302, headers={"location": "http://169.254.169.254/"}, content=b"x")

    with pytest.raises(MediaRejected):
        await adapter.download_media(
            media_job(adapter), media_store, "k", max_bytes=100, timeout_s=5
        )


@respx.mock
async def test_failed_buttons_prompt_does_not_fail_the_delivered_verdict(
    adapter: WhatsAppAdapter,
) -> None:
    route = respx.mock.post(MESSAGES).mock(
        side_effect=[
            httpx.Response(200, json={"messages": [{"id": "wamid.1"}]}),
            httpx.Response(500, json={"error": {"code": 1}}),
        ]
    )

    await adapter.send_verdict(job_for(adapter), "x" * 1500, buttons=BUTTONS)

    assert route.call_count == 2
