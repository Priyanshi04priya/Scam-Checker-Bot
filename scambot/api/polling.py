"""Long-polling runner for local development, feeding the same intake as the webhook."""

import argparse
import asyncio
import contextlib
from collections.abc import Sequence

from scambot.api.intake import open_telegram_intake
from scambot.channels.telegram import poll_updates
from scambot.core.logging import configure_logging, get_logger
from scambot.core.settings import get_settings


async def run_polling() -> None:
    settings = get_settings()
    async with open_telegram_intake(settings) as (intake, bot):
        get_logger().info("polling_started")
        await poll_updates(bot, intake.handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scambot.channels.telegram",
        description="Run the Telegram bot with long polling (local development only).",
    )
    parser.add_argument("--poll", action="store_true", help="receive updates by long polling")
    args = parser.parse_args(argv)
    if not args.poll:
        parser.error("nothing to do: pass --poll (deployments use the webhook instead)")
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.telegram_bot_token.get_secret_value():
        parser.error("TELEGRAM_BOT_TOKEN is not set")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_polling())
    return 0
