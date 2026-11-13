"""Webhook routes for each messaging platform."""

import json
import re
from typing import Annotated, Any, Final

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse

from scambot.api.intake import Intake, Outcome
from scambot.channels.whatsapp import normalise_number, verify_signature
from scambot.core.logging import get_logger
from scambot.core.security import secrets_match
from scambot.core.settings import Settings

# Header in which Telegram echoes the secret given to setWebhook.
TELEGRAM_AUTH_HEADER: Final = "X-Telegram-Bot-Api-Secret-Token"

WHATSAPP_MAX_BODY: Final = 1_048_576
WHATSAPP_MAX_MESSAGES: Final = 50
_CHALLENGE_RE: Final = re.compile(r"^[A-Za-z0-9_.~-]{1,128}$")

_log = get_logger(component="webhook")

router = APIRouter(prefix="/webhook")


@router.post("/telegram")
async def telegram_webhook(
    request: Request,
    token: Annotated[str | None, Header(alias=TELEGRAM_AUTH_HEADER)] = None,
) -> dict[str, bool]:
    """Receive one Telegram update. The secret is checked before the body is read."""
    settings: Settings = request.app.state.settings
    if not secrets_match(token, settings.telegram_webhook_secret):
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    try:
        update = json.loads(await request.body())
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST) from None
    if not isinstance(update, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST)
    intake: Intake = request.app.state.telegram_intake
    if await intake.handle(update) is Outcome.IN_PROGRESS:
        # Another delivery of this update is still being handled. A 5xx makes Telegram send it
        # again later, when it is either done (dropped) or released after a failure (handled).
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, headers={"Retry-After": "5"})
    return {"ok": True}


def _whatsapp_enabled(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    if not settings.whatsapp_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return settings


@router.get("/whatsapp")
async def whatsapp_verify(
    request: Request,
    mode: Annotated[str | None, Query(alias="hub.mode")] = None,
    verify_token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
    challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
) -> Response:
    """Answer Meta's subscription handshake by echoing the challenge."""
    settings = _whatsapp_enabled(request)
    if mode != "subscribe" or not secrets_match(verify_token, settings.whatsapp_verify_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    if challenge is None or not _CHALLENGE_RE.match(challenge):
        raise HTTPException(status.HTTP_400_BAD_REQUEST)
    return PlainTextResponse(challenge)


async def _capped_body(request: Request) -> bytes:
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > WHATSAPP_MAX_BODY:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > WHATSAPP_MAX_BODY:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE)
    return bytes(body)


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _whatsapp_messages(payload: dict[str, Any], phone_number_id: str) -> list[dict[str, Any]]:
    """One raw update per message: the message, its sender's number and profile name."""
    found: list[dict[str, Any]] = []
    for entry in _as_list(payload.get("entry")):
        for change in _as_list(entry.get("changes")) if isinstance(entry, dict) else []:
            if not isinstance(change, dict) or change.get("field") != "messages":
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            metadata = value.get("metadata") or {}
            if not isinstance(metadata, dict) or metadata.get("phone_number_id") != phone_number_id:
                _log.info("whatsapp_other_number_ignored")
                continue
            names = {
                normalise_number(str(c.get("wa_id", ""))): (
                    c["profile"].get("name") if isinstance(c.get("profile"), dict) else None
                )
                for c in _as_list(value.get("contacts"))
                if isinstance(c, dict)
            }
            for message in _as_list(value.get("messages")):
                if not isinstance(message, dict):
                    continue
                wa_id = normalise_number(str(message.get("from", "")))
                found.append({"message": message, "wa_id": wa_id, "profile_name": names.get(wa_id)})
    if len(found) > WHATSAPP_MAX_MESSAGES:
        _log.warning("whatsapp_messages_truncated", count=len(found))
    return found[:WHATSAPP_MAX_MESSAGES]


@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    signature: Annotated[str | None, Header(alias="X-Hub-Signature-256")] = None,
) -> dict[str, bool]:
    """Receive WhatsApp messages. The signature is checked on the raw bytes before parsing."""
    settings = _whatsapp_enabled(request)
    body = await _capped_body(request)
    if not verify_signature(body, signature, settings.whatsapp_app_secret):
        _log.warning("whatsapp_signature_invalid")
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST) from None
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST)
    if payload.get("object") != "whatsapp_business_account":
        return {"ok": True}
    intake: Intake | None = request.app.state.whatsapp_intake
    if intake is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE)
    busy = False
    for raw in _whatsapp_messages(payload, settings.whatsapp_phone_number_id):
        if await intake.handle(raw) is Outcome.IN_PROGRESS:
            busy = True
    if busy:
        # Meta sends the payload again; messages already finished are dropped by their claims.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, headers={"Retry-After": "5"})
    return {"ok": True}
