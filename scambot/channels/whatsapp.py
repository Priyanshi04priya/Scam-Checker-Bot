"""WhatsApp Cloud API adapter: webhook parsing, replies and media download."""

import hashlib
import hmac
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from scambot.channels.base import Button, parse_callback_data
from scambot.core.contracts import (
    Callback,
    Channel,
    Command,
    Job,
    JobKind,
    MediaMeta,
    MediaSource,
    Rejection,
)
from scambot.core.locales import Locales, load_locales
from scambot.core.logging import get_logger
from scambot.core.media import (
    ACCEPTED_MIME,
    MediaRejected,
    MediaUnavailable,
    kind_for_mime,
    normalise_mime,
)
from scambot.core.media_store import ORIGINAL, MediaStore
from scambot.core.security import user_hash

CHANNEL: Final[Channel] = "whatsapp"
TEXT_LIMIT: Final = 4096
INTERACTIVE_BODY_LIMIT: Final = 1024
BUTTON_TITLE_LIMIT: Final = 20
MAX_BUTTONS: Final = 3
SMALL_TALK_MAX_WORDS: Final = 4
MAX_LANGUAGE_COMMAND_CHARS: Final = 40
CHUNK_BYTES: Final = 65536
# Meta error codes for a reply outside the 24 hour window or to a recipient that cannot get it.
DROPPED_CODES: Final = frozenset({131047, 131030, 131026})
MEDIA_HOST_SUFFIXES: Final = (".fbsbx.com", ".facebook.com", ".fbcdn.net", ".whatsapp.net")

_log = get_logger(component="whatsapp")

_SIGNATURE_RE: Final = re.compile(r"^sha256=([0-9a-fA-F]{64})$")
_MEDIA_ID_RE: Final = re.compile(r"^\d{1,32}$")
_SLASH_RE: Final = re.compile(r"(?<![\w/])/(start|help|language|history|privacy|delete)\b")
_LANGUAGE_RE: Final = re.compile(r"^language(?:\s+(\S.*))?$", re.DOTALL)
_URL_RE: Final = re.compile(
    r"https?://|www\.|\b[\w-]+\.(?:com|in|org|net|co|io|me|ly|link)\b", re.I
)
_DIGIT_RUN_RE: Final = re.compile(r"\d{6,}")
_PUNCTUATION: Final = " \t\r\n.!?,;:'\"-"
_PLAIN_COMMANDS: Final = {
    "start": "start",
    "hi": "start",
    "hello": "start",
    "hey": "start",
    "namaste": "start",
    "help": "help",
    "menu": "help",
    "privacy": "privacy",
    "delete": "delete",
    "history": "history",
}
_IGNORED_TYPES: Final = frozenset(
    {"reaction", "system", "unsupported", "request_welcome", "ephemeral", "unknown"}
)


class WhatsAppError(Exception):
    """A Graph API call failed. The message is a short fixed code: never a URL or a token."""

    def __init__(self, code: int | None) -> None:
        super().__init__(f"whatsapp_error code={code}")
        self.code = code


def whatsapp_client() -> httpx.AsyncClient:
    """HTTP client for the Graph API: never follows redirects, with bounded waits."""
    return httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(20.0, connect=5.0),
    )


def verify_signature(raw_body: bytes, header: str | None, app_secret: SecretStr) -> bool:
    """True when ``header`` (``sha256=<hex>``) is the HMAC of ``raw_body`` under the app secret."""
    secret = app_secret.get_secret_value()
    if not secret or header is None:
        return False
    match = _SIGNATURE_RE.match(header)
    if match is None:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(match.group(1).lower().encode(), expected.encode())


def normalise_number(value: str) -> str:
    """Digits only: ``+91 98000 00001`` becomes ``919800000001``."""
    return re.sub(r"\D", "", value)


def _split_text(text: str, limit: int = TEXT_LIMIT) -> list[str]:
    """Pieces of at most ``limit`` characters, cut at paragraph boundaries where possible."""
    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = paragraph
        while len(current) > limit:
            parts.append(current[:limit])
            current = current[limit:]
    if current:
        parts.append(current)
    return parts or [""]


def _title(label: str) -> str:
    return label if len(label) <= BUTTON_TITLE_LIMIT else label[: BUTTON_TITLE_LIMIT - 1] + "…"


def _is_small_talk(text: str) -> bool:
    return (
        len(text.split()) < SMALL_TALK_MAX_WORDS
        and _URL_RE.search(text) is None
        and _DIGIT_RUN_RE.search(text) is None
    )


def _plain_command(text: str) -> tuple[str, str] | None:
    """The command name and arguments when the whole message is a plain-word command."""
    word = text.strip(_PUNCTUATION).removeprefix("/").strip(_PUNCTUATION).lower()
    if word in _PLAIN_COMMANDS:
        return _PLAIN_COMMANDS[word], ""
    match = _LANGUAGE_RE.match(word)
    if match is not None and len(word) <= MAX_LANGUAGE_COMMAND_CHARS:
        return "language", (match.group(1) or "").strip()
    return None


class WhatsAppAdapter:
    """WhatsApp Cloud API adapter on plain ``httpx``."""

    def __init__(  # noqa: PLR0913
        self,
        client: httpx.AsyncClient,
        *,
        token: SecretStr,
        phone_number_id: str,
        api_version: str,
        graph_url: str,
        allowlist: Iterable[str],
        user_id_pepper: SecretStr,
        mark_read: bool = True,
        locales: Locales | None = None,
    ) -> None:
        self._client = client
        self._token = token
        self._base = f"{graph_url.rstrip('/')}/{api_version}"
        self._messages_url = f"{self._base}/{phone_number_id}/messages"
        self._allowed = frozenset(n for n in map(normalise_number, allowlist) if n)
        self._pepper = user_id_pepper
        self._mark_read = mark_read
        self._locales = locales or load_locales()

    @property
    def channel(self) -> Channel:
        return CHANNEL

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token.get_secret_value()}"}

    # ----------------------------------------------------------------- inbound

    def parse_update(  # noqa: PLR0911
        self, raw: Mapping[str, Any], *, trace_id: str
    ) -> Job | Command | Rejection | Callback | None:
        """Turn one message (as split out by the webhook) into a job, command or button press."""
        message = raw.get("message")
        if not isinstance(message, dict):
            return None
        wamid = message.get("id")
        sender = message.get("from")
        kind = message.get("type")
        if not isinstance(wamid, str) or not isinstance(sender, str) or not isinstance(kind, str):
            return None
        wa_id = normalise_number(sender)
        if not wa_id:
            return None
        hashed = user_hash(wa_id, self._pepper)
        if wa_id not in self._allowed:
            _log.info("whatsapp_sender_not_allowed", sender=hashed[:12])
            return None
        context = message.get("context")
        forwarded = isinstance(context, dict) and context.get("forwarded") is True
        common: dict[str, Any] = {
            "trace_id": trace_id,
            "channel": CHANNEL,
            "chat_id": wa_id,
            "user_hash": hashed,
        }

        if kind == "interactive":
            return self._button_reply(message, wamid, common)
        if kind == "text":
            body = (message.get("text") or {}).get("body")
            if not isinstance(body, str) or not body.strip():
                return None
            if not forwarded:
                command = _plain_command(body)
                if command is None and _is_small_talk(body):
                    command = ("help", "")
                if command is not None:
                    return Command(
                        command_id=f"{CHANNEL}:{wamid}",
                        name=command[0],
                        args=command[1],
                        **common,
                    )
            return self._job(
                message,
                wamid,
                common,
                kind="text",
                text=body,
                media_ref=None,
                media=None,
                forwarded=forwarded,
            )

        if kind in {"image", "video", "audio", "document"}:
            return self._media_message(message, kind, wamid, common, forwarded)
        if kind in _IGNORED_TYPES:
            return None
        return Rejection(rejection_id=f"{CHANNEL}:{wamid}", reason="unsupported_type", **common)

    def _button_reply(
        self, message: dict[str, Any], wamid: str, common: dict[str, Any]
    ) -> Callback | None:
        interactive = message.get("interactive")
        if not isinstance(interactive, dict) or interactive.get("type") != "button_reply":
            return None
        data = (interactive.get("button_reply") or {}).get("id")
        parsed = parse_callback_data(data) if isinstance(data, str) else None
        if parsed is None:
            _log.info("callback_ignored", reason="data")
            return None
        return Callback(
            callback_id=f"{CHANNEL}:cb:{wamid}",
            query_id=wamid,
            action=parsed[0],
            value=parsed[1],
            **common,
        )

    def _media_message(
        self,
        message: dict[str, Any],
        kind: str,
        wamid: str,
        common: dict[str, Any],
        forwarded: bool,
    ) -> Job | Rejection | None:
        item = message.get(kind)
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return None
        mime = item.get("mime_type") if isinstance(item.get("mime_type"), str) else None
        caption = item.get("caption") if isinstance(item.get("caption"), str) else None
        source: MediaSource
        job_kind: JobKind | None
        if kind == "image":
            job_kind, source = "image", "photo"
        elif kind == "video":
            job_kind, source = "video", "video"
        elif kind == "audio":
            job_kind, source = (
                ("voice", "voice") if item.get("voice") is True else ("audio", "audio")
            )
        else:
            job_kind, source = kind_for_mime(mime), "document"
        if job_kind is None:
            return Rejection(rejection_id=f"{CHANNEL}:{wamid}", reason="unsupported_type", **common)
        meta = MediaMeta(source=source, mime_type=mime)
        return self._job(
            message,
            wamid,
            common,
            kind=job_kind,
            text=caption,
            media_ref=item["id"],
            media=meta,
            forwarded=forwarded,
        )

    def _job(  # noqa: PLR0913
        self,
        message: dict[str, Any],
        wamid: str,
        common: dict[str, Any],
        *,
        kind: JobKind,
        text: str | None,
        media_ref: str | None,
        media: MediaMeta | None,
        forwarded: bool,
    ) -> Job:
        context = message.get("context") or {}
        forward_meta: dict[str, Any] = (
            {
                "type": "forwarded",
                "frequently_forwarded": context.get("frequently_forwarded") is True,
            }
            if forwarded
            else {}
        )
        return Job(
            job_id=f"{CHANNEL}:{wamid}",
            kind=kind,
            text=text,
            media_ref=media_ref,
            media=media,
            forward_meta=forward_meta,
            preferred_lang=None,
            received_at=_received_at(message),
            **common,
        )

    # ------------------------------------------------------------------- media

    async def download_media(
        self, job: Job, store: MediaStore, key: str, *, max_bytes: int, timeout_s: float
    ) -> Path:
        media_id = job.media_ref or ""
        if not _MEDIA_ID_RE.match(media_id):
            raise MediaRejected("unreadable")
        info = await self._media_info(media_id, timeout_s)
        size = info.get("file_size")
        if isinstance(size, int) and size > max_bytes:
            raise MediaRejected("too_large")
        mime = info.get("mime_type")
        if isinstance(mime, str) and normalise_mime(mime) not in ACCEPTED_MIME[job.kind]:
            raise MediaRejected("unsupported_type")
        url = info.get("url")
        if not isinstance(url, str) or not _media_url_ok(url):
            raise MediaRejected("unreadable")
        try:
            async with self._client.stream(
                "GET", url, headers=self._auth(), timeout=timeout_s
            ) as response:
                if response.status_code >= 500 or response.status_code == 429:
                    raise MediaUnavailable(f"HTTP{response.status_code}")
                if response.status_code != httpx.codes.OK:
                    # Includes redirects, which are never followed.
                    raise MediaRejected("unreadable")
                length = response.headers.get("content-length")
                if length is not None and length.isdigit() and int(length) > max_bytes:
                    raise MediaRejected("too_large")
                return await store.put(
                    key, ORIGINAL, response.aiter_bytes(CHUNK_BYTES), max_bytes=max_bytes
                )
        except httpx.HTTPError as exc:
            raise MediaUnavailable(type(exc).__name__) from None

    async def _media_info(self, media_id: str, timeout_s: float) -> dict[str, Any]:
        try:
            response = await self._client.get(
                f"{self._base}/{media_id}", headers=self._auth(), timeout=timeout_s
            )
        except httpx.HTTPError as exc:
            raise MediaUnavailable(type(exc).__name__) from None
        if response.status_code == 429 or response.status_code >= 500:
            raise MediaUnavailable(f"HTTP{response.status_code}")
        if response.status_code >= 400:
            raise MediaRejected("unreadable")
        try:
            info = response.json()
        except ValueError:
            raise MediaRejected("unreadable") from None
        if not isinstance(info, dict):
            raise MediaRejected("unreadable")
        return info

    # ------------------------------------------------------------------- sends

    async def send_ack(self, job: Job, text: str) -> str:
        wamid = _wamid(job)
        if self._mark_read:
            await self._mark_as_read(wamid)
        sent = await self._send_parts(job.chat_id, text, wamid)
        return sent or "dropped"

    async def send_verdict(self, job: Job, text: str, *, buttons: Sequence[Button] = ()) -> None:
        lang = job.user_lang or job.preferred_lang
        await self._deliver(job.chat_id, text, buttons, lang, _wamid(job))

    async def send_text(
        self, chat_id: str, text: str, *, buttons: Sequence[Sequence[Button]] = ()
    ) -> None:
        flat = [b for row in buttons for b in row]
        await self._deliver(chat_id, text, flat, None, None)

    async def answer_callback(self, callback: Callback, text: str | None = None) -> None:
        if text:
            await self._send_parts(callback.chat_id, text, None)

    async def _deliver(
        self,
        to: str,
        text: str,
        buttons: Sequence[Button],
        lang: str | None,
        quote: str | None,
    ) -> None:
        rows = [
            {"type": "reply", "reply": {"id": b.data, "title": _title(b.label)}}
            for b in buttons[:MAX_BUTTONS]
        ]
        if not rows:
            await self._send_parts(to, text, quote)
        elif len(text) <= INTERACTIVE_BODY_LIMIT:
            await self._post(to, _interactive(_SLASH_RE.sub(r"\1", text), rows), quote)
        else:
            await self._send_parts(to, text, quote)
            prompt = self._locales.get(lang).buttons.prompt
            try:
                await self._post(to, _interactive(prompt, rows), None)
            except WhatsAppError as exc:
                # The verdict is out: raising would make a retry send it twice.
                _log.warning("whatsapp_prompt_failed", code=exc.code)

    async def _send_parts(self, to: str, text: str, quote: str | None) -> str | None:
        first: str | None = None
        for index, part in enumerate(_split_text(_SLASH_RE.sub(r"\1", text))):
            payload = {"type": "text", "text": {"body": part, "preview_url": False}}
            sent = await self._post(to, payload, quote if index == 0 else None)
            first = first or sent
        return first

    async def _post(self, to: str, content: dict[str, Any], quote: str | None) -> str | None:
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            **content,
        }
        if quote:
            payload["context"] = {"message_id": quote}
        try:
            response = await self._client.post(
                self._messages_url, json=payload, headers=self._auth()
            )
        except httpx.HTTPError:
            raise WhatsAppError(None) from None
        if response.is_success:
            return _first_message_id(response)
        code, kind = _error_of(response)
        if code in DROPPED_CODES:
            _log.warning("whatsapp_send_dropped", code=code)
            return None
        _log.warning("whatsapp_send_failed", status=response.status_code, code=code, type=kind)
        raise WhatsAppError(code or response.status_code)

    async def _mark_as_read(self, wamid: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": wamid,
            "typing_indicator": {"type": "text"},
        }
        try:
            response = await self._client.post(
                self._messages_url, json=payload, headers=self._auth()
            )
        except httpx.HTTPError as exc:
            _log.info("whatsapp_mark_read_failed", error=type(exc).__name__)
            return
        if not response.is_success:
            _log.info("whatsapp_mark_read_failed", status=response.status_code)


def _interactive(body: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "interactive",
        "interactive": {"type": "button", "body": {"text": body}, "action": {"buttons": rows}},
    }


def _wamid(job: Job) -> str:
    return job.job_id.removeprefix(f"{CHANNEL}:")


def _received_at(message: Mapping[str, Any]) -> datetime:
    stamp = message.get("timestamp")
    try:
        return datetime.fromtimestamp(int(str(stamp)), UTC)
    except (ValueError, OverflowError, OSError):
        return datetime.now(UTC)


def _media_url_ok(url: str) -> bool:
    parts = urlsplit(url)
    host = parts.hostname or ""
    return (
        parts.scheme == "https"
        and "@" not in parts.netloc
        and any(host.endswith(suffix) for suffix in MEDIA_HOST_SUFFIXES)
    )


def _first_message_id(response: httpx.Response) -> str | None:
    try:
        messages = response.json().get("messages")
        return str(messages[0]["id"])
    except (ValueError, AttributeError, LookupError, TypeError):
        return None


def _error_of(response: httpx.Response) -> tuple[int | None, str | None]:
    """Only the numeric code and type of a Graph error: its message may echo user content."""
    try:
        error = response.json().get("error") or {}
        code = error.get("code")
        kind = error.get("type")
    except (ValueError, AttributeError):
        return None, None
    return (code if isinstance(code, int) else None, kind if isinstance(kind, str) else None)
