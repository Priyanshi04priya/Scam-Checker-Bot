"""Scheduled maintenance jobs: the daily fact-check ingest and the media sweep."""

import asyncio
import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Any, Final

import httpx
from arq import cron
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncEngine

from scambot.api.admin.queries import StatsStore
from scambot.channels.base import ChannelAdapter
from scambot.core.config import AlertThresholds, AppConfig
from scambot.core.db import create_session_factory
from scambot.core.logging import get_logger
from scambot.core.media_store import MediaStore
from scambot.core.runtime import SettingsStore, effective_sources
from scambot.core.settings import Settings
from scambot.core.store import RequestStore
from scambot.factcheck_index.embed import create_embedder
from scambot.factcheck_index.ingest import (
    USER_AGENT,
    IngestDeps,
    IngestOptions,
    PageFetcher,
    PublisherReviews,
    SourceReport,
    run_ingest,
)
from scambot.factcheck_index.search import PgFactcheckIndex
from scambot.mcp_server.clients.factcheck_tools import FactCheckToolsClient

INGEST_KEY: Final = "factcheck_ingest"
INGEST_HTTP_KEY: Final = "factcheck_ingest_http"
SETTINGS_STORE_KEY: Final = "settings_store"
INGEST_TIMEOUT_S: Final = 1800
MEDIA_SWEEP_KEY: Final = "media_sweep"

_log = get_logger(component="cron")


def build_ingest_deps(
    settings: Settings, config: AppConfig, engine: AsyncEngine
) -> tuple[IngestDeps | None, httpx.AsyncClient | None]:
    """Ingest dependencies plus the HTTP client to close; (None, None) without an embedder."""
    embedder = create_embedder(settings)
    if embedder is None:
        return None, None
    limits = config.thresholds.ingest
    http = httpx.AsyncClient(
        timeout=limits.fetch_timeout_s,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
    )
    reviews: PublisherReviews | None = None
    if settings.google_api_key.get_secret_value():
        reviews = FactCheckToolsClient(http, settings.google_api_key)
    deps = IngestDeps(
        catalogue=config.sources,
        limits=limits,
        index=PgFactcheckIndex(create_session_factory(engine)),
        embedder=embedder,
        fetcher=PageFetcher(http, limits),
        reviews=reviews,
    )
    return deps, http


def totals(reports: list[SourceReport]) -> dict[str, int]:
    """Counts summed over every source."""
    names = ("api_items", "feed_items", "fetched", "skipped", "failed", "upserted")
    return {name: sum(getattr(report, name) for report in reports) for name in names}


async def ingest_factchecks(ctx: dict[str, Any]) -> dict[str, int]:
    """Index recent fact-checks from every configured publisher."""
    deps: IngestDeps | None = ctx.get(INGEST_KEY)
    if deps is None:
        _log.info("ingest_skipped", reason="no_embedder")
        return {}
    store: SettingsStore | None = ctx.get(SETTINGS_STORE_KEY)
    if store is not None:
        # Read at every run, so publishers changed on the admin page apply the next night.
        deps = dataclasses.replace(deps, catalogue=await effective_sources(store, deps.catalogue))
    options = IngestOptions(
        lookback_days=deps.limits.lookback_days, max_api_pages=deps.limits.max_api_pages
    )
    reports = await run_ingest(deps, options)
    summary = totals(reports)
    failed = [report.source for report in reports if report.error]
    _log.info("ingest_done", failed_sources=failed, **summary)
    return summary


INGEST_CRON: Final = cron(
    ingest_factchecks,
    name="ingest_factchecks",
    hour={2},
    minute={0},
    run_at_startup=False,
    unique=True,
    timeout=INGEST_TIMEOUT_S,
    max_tries=1,
)


@dataclass(frozen=True)
class SweepDeps:
    store: MediaStore
    older_than: dt.timedelta


async def sweep_media(ctx: dict[str, Any]) -> int:
    """Delete downloaded media that a finished or crashed job left behind."""
    deps: SweepDeps = ctx[MEDIA_SWEEP_KEY]
    removed = await asyncio.to_thread(deps.store.sweep, deps.older_than)
    _log.info("media_swept", removed=removed)
    return removed


SWEEP_CRON: Final = cron(
    sweep_media,
    name="sweep_media",
    minute=set(range(0, 60, 10)),
    run_at_startup=True,
    unique=True,
    timeout=120,
    max_tries=1,
)


PURGE_KEY: Final = "retention_purge"


@dataclass(frozen=True)
class PurgeDeps:
    store: RequestStore
    days: int


async def purge_retention(ctx: dict[str, Any]) -> int:
    """Delete requests, and everything attached to them, older than the retention period."""
    deps: PurgeDeps = ctx[PURGE_KEY]
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=deps.days)
    removed = await deps.store.purge_older_than(cutoff)
    _log.info("retention_purged", removed=removed, days=deps.days)
    return removed


PURGE_CRON: Final = cron(
    purge_retention,
    name="purge_retention",
    hour={3},
    minute={30},
    run_at_startup=False,
    unique=True,
    timeout=600,
    max_tries=1,
)


ALERT_KEY: Final = "checker_alert"


@dataclass(frozen=True)
class AlertDeps:
    stats: StatsStore
    redis: ArqRedis
    adapter: ChannelAdapter
    chat_id: str
    limits: AlertThresholds
    names: dict[str, str] = dataclasses.field(default_factory=dict)


async def checker_error_alert(ctx: dict[str, Any]) -> list[str]:
    """Message the operator when a checker has been failing; once an hour per checker."""
    deps: AlertDeps = ctx[ALERT_KEY]
    if not deps.chat_id:
        _log.info("checker_alert_skipped", reason="no_chat")
        return []
    limits = deps.limits
    since = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=limits.window_min)
    alerted: list[str] = []
    for item in await deps.stats.checker_error_rates(since):
        if item.total < limits.min_runs or item.rate <= limits.error_rate:
            continue
        key = f"alert:{item.checker}"
        if not await deps.redis.set(key, 1, nx=True, ex=limits.dedupe_min * 60):
            continue
        text = (
            f"Alert: {deps.names.get(item.checker, item.checker)} failed {item.errors} of "
            f"{item.total} times in the last {limits.window_min} minutes "
            f"({round(item.rate * 100)}%). Top errors: {', '.join(item.top_errors) or 'none'}."
        )
        try:
            await deps.adapter.send_text(deps.chat_id, text)
        except Exception as exc:  # try again at the next run instead of staying silent
            _log.warning("checker_alert_failed", error=type(exc).__name__)
            await deps.redis.delete(key)
            continue
        alerted.append(item.checker)
    _log.info("checker_alert_done", alerted=alerted)
    return alerted


ALERT_CRON: Final = cron(
    checker_error_alert,
    name="checker_error_alert",
    minute=set(range(0, 60, 5)),
    run_at_startup=False,
    unique=True,
    timeout=60,
    max_tries=1,
)
