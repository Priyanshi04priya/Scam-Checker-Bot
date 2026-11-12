"""Tests for the backfill command and the scheduled ingest job (fakes only)."""

from typing import Any

import httpx
import pytest

from scambot.core.config import AppConfig
from scambot.core.db import create_engine
from scambot.core.settings import Settings
from scambot.factcheck_index.embed import SentenceTransformerEmbedder
from scambot.factcheck_index.ingest import IngestDeps, IngestOptions, PageFetcher, SourceReport
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from scambot.mcp_server.clients.factcheck_tools import FactCheckToolsClient
from scambot.worker import backfill, cron
from tests.fakes import KeywordEmbedder


def fake_deps(app_config: AppConfig, http: httpx.AsyncClient) -> IngestDeps:
    limits = app_config.thresholds.ingest
    return IngestDeps(
        catalogue=app_config.sources,
        limits=limits,
        index=InMemoryFactcheckIndex(),
        embedder=KeywordEmbedder(),
        fetcher=PageFetcher(http, limits),
        reviews=None,
    )


def test_arguments() -> None:
    defaults = backfill.parse_args([])
    assert (defaults.lookback_days, defaults.max_api_pages, defaults.sources) == (730, 20, None)

    args = backfill.parse_args(
        ["--lookback-days", "7", "--max-api-pages", "2", "--source", "altnews", "--source", "boom"]
    )
    assert (args.lookback_days, args.max_api_pages, args.sources) == (7, 2, ["altnews", "boom"])


def test_backfill_runs_and_closes(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[IngestOptions] = []
    clients: list[httpx.AsyncClient] = []

    def build(*_args: Any) -> tuple[IngestDeps, httpx.AsyncClient]:
        http = httpx.AsyncClient()
        clients.append(http)
        return fake_deps(app_config, http), http

    async def run(_deps: IngestDeps, options: IngestOptions) -> list[SourceReport]:
        seen.append(options)
        return [SourceReport(source="altnews", upserted=3)]

    monkeypatch.setattr(cron, "build_ingest_deps", build)
    monkeypatch.setattr(backfill, "run_ingest", run)

    assert backfill.main(["--lookback-days", "7", "--source", "altnews"]) == 0

    (options,) = seen
    assert options == IngestOptions(lookback_days=7, max_api_pages=20, only=frozenset({"altnews"}))
    assert clients[0].is_closed


def test_backfill_without_an_embedder_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cron, "build_ingest_deps", lambda *_args: (None, None))

    assert backfill.main([]) == 2


async def test_cron_job_without_deps_is_skipped() -> None:
    assert await cron.ingest_factchecks({}) == {}
    assert await cron.ingest_factchecks({cron.INGEST_KEY: None}) == {}


async def test_cron_job_returns_totals(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[IngestOptions] = []

    async def run(_deps: IngestDeps, options: IngestOptions) -> list[SourceReport]:
        seen.append(options)
        return [
            SourceReport(source="altnews", api_items=2, upserted=2),
            SourceReport(source="boom", feed_items=1, failed=1, error="ConnectError"),
        ]

    monkeypatch.setattr(cron, "run_ingest", run)
    async with httpx.AsyncClient() as http:
        totals = await cron.ingest_factchecks({cron.INGEST_KEY: fake_deps(app_config, http)})

    assert totals == {
        "api_items": 2,
        "feed_items": 1,
        "fetched": 0,
        "skipped": 0,
        "failed": 1,
        "upserted": 2,
    }
    limits = app_config.thresholds.ingest
    assert seen == [
        IngestOptions(lookback_days=limits.lookback_days, max_api_pages=limits.max_api_pages)
    ]


async def test_build_ingest_deps(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    engine = create_engine(settings.database_url)
    try:
        monkeypatch.setattr(cron, "create_embedder", lambda _s: None)
        assert cron.build_ingest_deps(settings, app_config, engine) == (None, None)

        monkeypatch.setattr(cron, "create_embedder", lambda _s: SentenceTransformerEmbedder("m"))
        deps, http = cron.build_ingest_deps(settings, app_config, engine)
        assert deps is not None
        assert http is not None
        assert deps.reviews is None
        assert http.headers["User-Agent"] == "scambot-factcheck-indexer/0.1"
        await http.aclose()

        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        deps, http = cron.build_ingest_deps(Settings(_env_file=None), app_config, engine)
        assert deps is not None
        assert isinstance(deps.reviews, FactCheckToolsClient)
        assert http is not None
        await http.aclose()
    finally:
        await engine.dispose()
