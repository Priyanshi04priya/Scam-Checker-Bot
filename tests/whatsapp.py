"""Builders for WhatsApp Cloud API webhook payloads, signed the way Meta signs them."""

import hashlib
import hmac
import json
from typing import Any

from pydantic import SecretStr

from scambot.core.settings import Settings

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"
ACCESS_TOKEN = "EAAB-test-access-token"
PHONE_ID = "1055550001"
SENDER = "919800000001"
OTHER_SENDER = "919800000002"
STRANGER = "919800000009"
GRAPH = "https://graph.facebook.com/v23.0"
CDN = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=1&ext=2&hash=abc"


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "whatsapp_token": SecretStr(ACCESS_TOKEN),
        "whatsapp_app_secret": SecretStr(APP_SECRET),
        "whatsapp_verify_token": SecretStr(VERIFY_TOKEN),
        "whatsapp_phone_number_id": PHONE_ID,
        "whatsapp_allowlist": (SENDER, OTHER_SENDER),
    }
    return Settings(_env_file=None, **{**values, **overrides})


def message(
    kind: str, body: dict[str, Any] | None = None, *, wamid: str = "wamid.A1", **extra: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "from": extra.pop("sender", SENDER),
        "id": wamid,
        "timestamp": "1760000000",
        "type": kind,
    }
    if body is not None:
        base[kind] = body
    return {**base, **extra}


def text(body: str, **kwargs: Any) -> dict[str, Any]:
    return message("text", {"body": body}, **kwargs)


def button(data: str, **kwargs: Any) -> dict[str, Any]:
    reply = {"type": "button_reply", "button_reply": {"id": data, "title": "x"}}
    return message("interactive", reply, **kwargs) | {"interactive": reply}


def raw(msg: dict[str, Any]) -> dict[str, Any]:
    """What the webhook hands to the adapter for one message."""
    return {"message": msg, "wa_id": msg["from"], "profile_name": "Asha"}


def payload(*messages: dict[str, Any], phone_id: str = PHONE_ID) -> dict[str, Any]:
    contacts = [{"wa_id": m["from"], "profile": {"name": "Asha"}} for m in messages]
    value: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15550000000", "phone_number_id": phone_id},
        "contacts": contacts,
        "messages": list(messages),
    }
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "1", "changes": [{"field": "messages", "value": value}]}],
    }


def status_payload() -> dict[str, Any]:
    value = {
        "messaging_product": "whatsapp",
        "metadata": {"phone_number_id": PHONE_ID},
        "statuses": [{"id": "wamid.A1", "status": "delivered", "recipient_id": SENDER}],
    }
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "1", "changes": [{"field": "messages", "value": value}]}],
    }


def sign(body: bytes, secret: str = APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def signed(data: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(data).encode()
    return body, {"X-Hub-Signature-256": sign(body), "Content-Type": "application/json"}
