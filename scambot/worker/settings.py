"""Queue worker configuration."""

import datetime as dt
from collections.abc import Sequence
from typing import Any

import httpx
from aiogram import Bot
from anthropic import AsyncAnthropic
from arq.connections import RedisSettings
from arq.cron import CronJob
from arq.typing import StartupShutdown, WorkerCoroutine
from arq.worker import Function, func
from sqlalchemy.ext.asyncio import AsyncEngine

from scambot.agent.executor import Executor, McpToolCaller
from scambot.channels.base import ChannelAdapter
from scambot.channels.telegram import TelegramAdapter, create_bot
from scambot.core.config import load_config
from scambot.core.db import create_engine, create_session_factory
from scambot.core.llm import create_llm_client, load_prompts
from scambot.core.locales import load_locales
from scambot.core.logging import configure_logging
from scambot.core.redis import PROCESS_JOB, connection_settings
from scambot.core.settings import get_settings
from scambot.core.store import SqlRequestStore
from scambot.preprocess.entities import HeadRedirectExpander, NoopExpander, UrlExpander
from scambot.worker.cron import INGEST_CRON, INGEST_HTTP_KEY, INGEST_KEY, IST, build_ingest_deps
from scambot.worker.pipeline import DEPS_KEY, PipelineDeps, process_job

BOT_KEY = "telegram_bot"
HTTP_KEY = "http_client"
ANTHROPIC_KEY = "anthropic_client"
ENGINE_KEY = "db_engine"

EXPAND_TIMEOUT_S = 3.0
# Time kept after the pipeline deadline for the partial reply, the store writes and delivery.
POST_DEADLINE_MARGIN_S = 10.0


async def startup(ctx: dict[str, Any]) -> None:
    """Create the long-lived clients once per worker process; invalid config stops the start."""
    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_config()
    if config.thresholds.pipeline.deadline_s + POST_DEADLINE_MARGIN_S > WorkerSettings.job_timeout:
        msg = (
            f"pipeline deadline_s must leave {POST_DEADLINE_MARGIN_S:g}s before the "
            f"{WorkerSettings.job_timeout}s job timeout"
        )
        raise ValueError(msg)
    prompts = load_prompts()
    locales = load_locales()
    bot = create_bot(settings.telegram_bot_token, settings.telegram_api_url)
    ctx[BOT_KEY] = bot
    adapters: dict[str, ChannelAdapter] = {
        "telegram": TelegramAdapter(bot, settings.user_id_pepper),
    }
    llm, anthropic_client = create_llm_client(settings, prompts)
    ctx[ANTHROPIC_KEY] = anthropic_client
    engine = create_engine(settings.database_url)
    ctx[ENGINE_KEY] = engine
    ctx[INGEST_KEY], ctx[INGEST_HTTP_KEY] = build_ingest_deps(settings, config, engine)
    expander: UrlExpander = NoopExpander()
    if settings.expand_short_urls:
        http = httpx.AsyncClient(timeout=EXPAND_TIMEOUT_S, follow_redirects=False)
        ctx[HTTP_KEY] = http
        expander = HeadRedirectExpander(
            http,
            frozenset(config.scam_patterns.shorteners),
            config.thresholds.entities.max_redirects,
        )
    caller = McpToolCaller(settings.mcp_url, config.thresholds.pipeline.mcp_connect_timeout_s)
    ctx[DEPS_KEY] = PipelineDeps(
        adapters=adapters,
        locales=locales,
        config=config,
        llm=llm,
        executor=Executor(caller),
        store=SqlRequestStore(create_session_factory(engine)),
        expander=expander,
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    """Close whatever ``startup`` created; tolerant of a partial start."""
    bot: Bot | None = ctx.get(BOT_KEY)
    if bot is not None:
        await bot.session.close()
    for key in (HTTP_KEY, INGEST_HTTP_KEY):
        http: httpx.AsyncClient | None = ctx.get(key)
        if http is not None:
            await http.aclose()
    anthropic_client: AsyncAnthropic | None = ctx.get(ANTHROPIC_KEY)
    if anthropic_client is not None:
        await anthropic_client.close()
    engine: AsyncEngine | None = ctx.get(ENGINE_KEY)
    if engine is not None:
        await engine.dispose()


class WorkerSettings:
    """Run with ``arq scambot.worker.settings.WorkerSettings``."""

    functions: Sequence[WorkerCoroutine | Function] = (func(process_job, name=PROCESS_JOB),)
    cron_jobs: Sequence[CronJob] | None = (INGEST_CRON,)
    on_startup: StartupShutdown | None = startup
    on_shutdown: StartupShutdown | None = shutdown
    redis_settings: RedisSettings = connection_settings(get_settings().redis_url)
    job_timeout = 100
    max_tries = 3
    keep_result = 0
    max_jobs = 10
    timezone: dt.timezone = IST
