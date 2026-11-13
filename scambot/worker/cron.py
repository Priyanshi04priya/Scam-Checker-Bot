"""Scheduled maintenance jobs: the daily fact-check ingest and the media sweep."""

import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Any, Final

import httpx
from arq import cron
from sqlalchemy.ext.asyncio import AsyncEngine

from scambot.core.config import AppConfig
from scambot.core.db import create_session_factory
from scambot.core.logging import get_logger
from scambot.core.media_store import MediaStore
from scambot.core.settings import Settings
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
INGEST_TIMEOUT_S: Final = 1800
MEDIA_SWEEP_KEY: Final = "media_sweep"
# India Standard Time has no daylight saving, so a fixed offset is exact.
IST: Final = dt.timezone(dt.timedelta(hours=5, minutes=30), "IST")

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
