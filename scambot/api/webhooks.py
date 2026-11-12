"""Webhook routes for each messaging platform."""

import json
from typing import Annotated, Final

from fastapi import APIRouter, Header, HTTPException, Request, status

from scambot.api.intake import Intake, Outcome
from scambot.core.security import secrets_match
from scambot.core.settings import Settings

# Header in which Telegram echoes the secret given to setWebhook.
TELEGRAM_AUTH_HEADER: Final = "X-Telegram-Bot-Api-Secret-Token"

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
