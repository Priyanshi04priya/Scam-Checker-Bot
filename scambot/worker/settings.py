"""Queue worker configuration."""

from collections.abc import Sequence
from typing import Any

from aiogram import Bot
from arq.connections import RedisSettings
from arq.cron import CronJob
from arq.typing import StartupShutdown, WorkerCoroutine
from arq.worker import Function, func

from scambot.channels.base import ChannelAdapter
from scambot.channels.telegram import TelegramAdapter, create_bot
from scambot.core.locales import load_locales
from scambot.core.logging import configure_logging
from scambot.core.redis import PROCESS_JOB, connection_settings
from scambot.core.settings import get_settings
from scambot.worker.pipeline import ADAPTERS_KEY, LOCALES_KEY, process_job

BOT_KEY = "telegram_bot"


async def startup(ctx: dict[str, Any]) -> None:
    """Create the reply clients once per worker process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    bot = create_bot(settings.telegram_bot_token, settings.telegram_api_url)
    adapters: dict[str, ChannelAdapter] = {
        "telegram": TelegramAdapter(bot, settings.user_id_pepper),
    }
    ctx[BOT_KEY] = bot
    ctx[ADAPTERS_KEY] = adapters
    ctx[LOCALES_KEY] = load_locales()


async def shutdown(ctx: dict[str, Any]) -> None:
    bot: Bot | None = ctx.get(BOT_KEY)
    if bot is not None:
        await bot.session.close()


class WorkerSettings:
    """Run with ``arq scambot.worker.settings.WorkerSettings``."""

    functions: Sequence[WorkerCoroutine | Function] = (func(process_job, name=PROCESS_JOB),)
    cron_jobs: Sequence[CronJob] | None = None
    on_startup: StartupShutdown | None = startup
    on_shutdown: StartupShutdown | None = shutdown
    redis_settings: RedisSettings = connection_settings(get_settings().redis_url)
    job_timeout = 100
    max_tries = 3
    keep_result = 0
    max_jobs = 10
