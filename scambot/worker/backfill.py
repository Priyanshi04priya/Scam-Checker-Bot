"""One-off deep ingest of older fact-checks.

Run with ``python -m scambot.worker.backfill [--lookback-days 730] [--max-api-pages 20]
[--source ID ...]``.
"""

import argparse
import asyncio
from collections.abc import Sequence

from scambot.core.config import load_config
from scambot.core.db import create_engine
from scambot.core.logging import configure_logging, get_logger
from scambot.core.settings import get_settings
from scambot.factcheck_index.ingest import IngestOptions, run_ingest
from scambot.worker import cron

EXIT_UNAVAILABLE = 2

_log = get_logger(component="backfill")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index older fact-check articles.")
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--max-api-pages", type=int, default=20)
    parser.add_argument("--source", action="append", dest="sources", metavar="ID")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_config()
    engine = create_engine(settings.database_url)
    try:
        deps, http = cron.build_ingest_deps(settings, config, engine)
        if deps is None:
            _log.error("backfill_unavailable", reason="no_embedder")
            return EXIT_UNAVAILABLE
        try:
            options = IngestOptions(
                lookback_days=args.lookback_days,
                max_api_pages=args.max_api_pages,
                only=frozenset(args.sources) if args.sources else None,
            )
            reports = await run_ingest(deps, options)
        finally:
            if http is not None:
                await http.aclose()
        for report in reports:
            _log.info("backfill_source", **vars(report))
        _log.info("backfill_done", **cron.totals(reports))
        return 0
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
