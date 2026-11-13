"""Command line for the Telegram channel: local long polling and webhook management."""

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Sequence

from scambot.api.intake import open_telegram_intake
from scambot.api.webhook_setup import (
    delete_webhook,
    describe_error,
    set_webhook,
    validate_secret,
    validate_webhook_url,
    webhook_info,
)
from scambot.channels.telegram import create_bot, poll_updates
from scambot.core.logging import configure_logging, get_logger
from scambot.core.settings import get_settings


async def run_polling() -> None:
    settings = get_settings()
    async with open_telegram_intake(settings) as (intake, bot):
        get_logger().info("polling_started")
        await poll_updates(bot, intake.handle)


async def run_webhook_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    bot = create_bot(settings.telegram_bot_token, settings.telegram_api_url)
    try:
        if args.set_webhook:
            await set_webhook(
                bot, args.set_webhook, settings.telegram_webhook_secret.get_secret_value()
            )
            print("webhook registered")  # noqa: T201
        elif args.delete_webhook:
            await delete_webhook(bot, drop_pending=args.drop_pending)
            print("webhook removed")  # noqa: T201
        print(await webhook_info(bot))  # noqa: T201
    except Exception as exc:
        print(f"Telegram call failed: {describe_error(exc)}", file=sys.stderr)  # noqa: T201
        return 1
    finally:
        await bot.session.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scambot.channels.telegram",
        description=(
            "Manage how the Telegram bot receives updates: register or remove the production "
            "webhook, or run long polling (local development only; use a separate development "
            "bot, because polling removes the webhook of the bot it runs with)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--poll", action="store_true", help="receive updates by long polling")
    mode.add_argument("--set-webhook", metavar="URL", help="register URL as the webhook")
    mode.add_argument("--delete-webhook", action="store_true", help="remove the webhook")
    mode.add_argument("--webhook-info", action="store_true", help="show the current webhook")
    parser.add_argument(
        "--drop-pending", action="store_true", help="with --delete-webhook: discard queued updates"
    )
    args = parser.parse_args(argv)
    if args.drop_pending and not args.delete_webhook:
        parser.error("--drop-pending only goes with --delete-webhook")
    if not (args.poll or args.set_webhook or args.delete_webhook or args.webhook_info):
        parser.error(
            "nothing to do: pass --poll, --set-webhook, --delete-webhook or --webhook-info"
        )
    if args.set_webhook and (problem := validate_webhook_url(args.set_webhook)):
        parser.error(problem)
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.telegram_bot_token.get_secret_value():
        parser.error("TELEGRAM_BOT_TOKEN is not set")
    if args.set_webhook:
        secret = settings.telegram_webhook_secret.get_secret_value()
        if not secret:
            parser.error("TELEGRAM_WEBHOOK_SECRET is not set")
        if problem := validate_secret(secret):
            parser.error(problem)
    if args.poll:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_polling())
        return 0
    return asyncio.run(run_webhook_command(args))
