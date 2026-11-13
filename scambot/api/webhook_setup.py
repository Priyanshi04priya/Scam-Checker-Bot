"""Register, inspect and remove the Telegram webhook (run once after a deployment)."""

import re
from typing import Any, Protocol
from urllib.parse import urlsplit

from aiogram.exceptions import TelegramBadRequest, TelegramUnauthorizedError

from scambot.channels.telegram import ALLOWED_UPDATES

WEBHOOK_PATH = "/webhook/telegram"
_ALLOWED_PORTS = frozenset({80, 88, 443, 8443})
_SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


class WebhookBot(Protocol):
    """The part of the Bot API client that is used here."""

    async def set_webhook(
        self,
        url: str,
        *,
        max_connections: int | None = None,
        allowed_updates: list[str] | None = None,
        drop_pending_updates: bool | None = None,
        secret_token: str | None = None,
    ) -> Any: ...
    async def delete_webhook(self, *, drop_pending_updates: bool | None = None) -> Any: ...
    async def get_webhook_info(self) -> Any: ...


def validate_webhook_url(url: str) -> str | None:
    """Return a problem description, or ``None`` when ``url`` can be registered."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "the URL is not valid"
    if parts.scheme != "https" or not parts.hostname or "@" in parts.netloc:
        return "the webhook URL must be an https:// URL with a host name and no credentials"
    if parts.query or parts.fragment or url.endswith(("?", "#")):
        return "the webhook URL must not have a query string or fragment"
    if parts.path != WEBHOOK_PATH:
        return f"the webhook URL path must be exactly {WEBHOOK_PATH}"
    if port is not None and port not in _ALLOWED_PORTS:
        return "Telegram only delivers webhooks to ports 80, 88, 443 and 8443"
    return None


def validate_secret(secret: str) -> str | None:
    if not _SECRET_PATTERN.fullmatch(secret):
        return "TELEGRAM_WEBHOOK_SECRET must be 1-256 characters from A-Z, a-z, 0-9, _ and -"
    return None


async def set_webhook(bot: WebhookBot, url: str, secret: str) -> None:
    """Always register (it is idempotent and also rotates the secret, which Telegram never
    returns, so a comparison with the current state could not detect a changed secret)."""
    await bot.set_webhook(
        url=url,
        secret_token=secret,
        allowed_updates=ALLOWED_UPDATES,
        drop_pending_updates=False,
        max_connections=20,
    )


async def delete_webhook(bot: WebhookBot, *, drop_pending: bool) -> None:
    await bot.delete_webhook(drop_pending_updates=drop_pending)


async def webhook_info(bot: WebhookBot) -> str:
    """A one-per-line summary; it never contains the token or the secret."""
    info = await bot.get_webhook_info()
    return "\n".join(
        (
            f"url: {info.url or '(none)'}",
            f"pending_update_count: {info.pending_update_count}",
            f"last_error_message: {info.last_error_message or '(none)'}",
        )
    )


def describe_error(exc: Exception) -> str:
    """Safe text for a failed call: aiogram messages for network errors can echo the request
    URL, which contains the bot token, so only the type is shown for those."""
    if isinstance(exc, TelegramBadRequest | TelegramUnauthorizedError):
        return f"{type(exc).__name__}: {exc.message}"
    return type(exc).__name__
