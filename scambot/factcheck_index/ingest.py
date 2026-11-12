"""Fetches fact-check articles from publisher feeds and the Fact Check Tools search, and stores
them in the index.

Only pages on a publisher's own site are fetched, robots.txt is honoured, every host gets a
pause between requests and page bodies are size-capped.
"""

import asyncio
import calendar
import datetime as dt
import io
import json
import re
import time
import urllib.robotparser
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol
from urllib.parse import urljoin, urlsplit

import feedparser
import httpx
import trafilatura

from scambot.core.config import FactcheckSource, IngestThresholds, SourcesConfig, host_on_site
from scambot.core.logging import get_logger
from scambot.factcheck_index.embed import Embedder
from scambot.factcheck_index.records import (
    MAX_CLAIM_CHARS,
    MAX_RATING_CHARS,
    MAX_TITLE_CHARS,
    MAX_URL_CHARS,
    ArticleRecord,
    ClaimReview,
    canonical_url,
    normalise_language,
    passage_text,
    record_from_review,
)
from scambot.factcheck_index.search import EmbeddedArticle, FactcheckIndex

USER_AGENT: Final = "scambot-factcheck-indexer/0.1"
MAX_REDIRECTS: Final = 3
MAX_JSON_LD_BLOCKS: Final = 20

_JSON_LD_RE: Final = re.compile(
    r"<script[^>]*type\s*=\s*[\"']application/ld\+json[\"'][^>]*>(.*?)</script\s*>",
    re.IGNORECASE | re.DOTALL,
)

_log = get_logger(component="ingest")


class PublisherReviews(Protocol):
    async def by_publisher(
        self, site: str, *, max_age_days: int, page_size: int, max_pages: int
    ) -> list[ClaimReview]: ...


@dataclass(frozen=True)
class FeedEntry:
    url: str
    title: str
    published_at: dt.datetime | None


@dataclass(frozen=True)
class ClaimReviewMarkup:
    claim: str | None
    rating: str | None
    published_at: dt.datetime | None
    language: str | None
    url: str | None


# --- parsing -------------------------------------------------------------------------------


def _struct_time(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.fromtimestamp(calendar.timegm(value), tz=dt.UTC)


def parse_feed(data: bytes, max_items: int) -> list[FeedEntry]:
    """Entries with a link and a title, in feed order (bytes, so feedparser never fetches)."""
    parsed = feedparser.parse(io.BytesIO(data))
    entries: list[FeedEntry] = []
    for entry in parsed.entries:
        link = str(entry.get("link") or "").strip()
        title = str(entry.get("title") or "").strip()
        if not link or not title:
            continue
        published = _struct_time(entry.get("published_parsed") or entry.get("updated_parsed"))
        entries.append(FeedEntry(url=link, title=title, published_at=published))
        if len(entries) >= max_items:
            break
    return entries


def _parse_date(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _walk(node: object) -> Iterator[dict[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
    elif isinstance(node, dict):
        yield node
        for key in ("@graph", "mainEntity", "review", "itemReviewed"):
            if key in node:
                yield from _walk(node[key])


def _is_claim_review(node: dict[str, Any]) -> bool:
    kind = node.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    return "ClaimReview" in kinds


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _rating(value: object) -> str | None:
    if isinstance(value, dict):
        return _text(value.get("alternateName")) or _text(value.get("name"))
    return None


def _json_ld_blocks(html: str) -> Iterator[object]:
    for index, match in enumerate(_JSON_LD_RE.finditer(html)):
        if index >= MAX_JSON_LD_BLOCKS:
            return
        try:
            yield json.loads(match.group(1), strict=False)
        except ValueError:
            continue


def extract_claim_review(html: str) -> ClaimReviewMarkup | None:
    """The first schema.org ``ClaimReview`` in the page's JSON-LD, if any."""
    for block in _json_ld_blocks(html):
        for node in _walk(block):
            if not _is_claim_review(node):
                continue
            return ClaimReviewMarkup(
                claim=_text(node.get("claimReviewed")),
                rating=_rating(node.get("reviewRating")),
                published_at=_parse_date(node.get("datePublished")),
                language=normalise_language(node.get("inLanguage")),
                url=_text(node.get("url")),
            )
    return None


def _metadata(html: str, url: str) -> tuple[str | None, dt.datetime | None]:
    document = trafilatura.extract_metadata(html, default_url=url)
    title = _text(document.title)
    return title, _parse_date(document.date)


def article_from_page(source: FactcheckSource, entry: FeedEntry, html: str) -> ArticleRecord:
    """The record for a feed entry's page: ClaimReview markup first, then the feed and the
    page's own metadata."""
    markup = extract_claim_review(html)
    meta_title, meta_date = _metadata(html, entry.url)
    claim = (markup.claim if markup else None) or entry.title
    published = (markup.published_at if markup else None) or entry.published_at or meta_date
    rating = markup.rating if markup else None
    return ArticleRecord(
        source=source.id,
        url=entry.url,
        title=(meta_title or entry.title)[:MAX_TITLE_CHARS],
        claim=claim[:MAX_CLAIM_CHARS],
        rating=rating[:MAX_RATING_CHARS] if rating else None,
        published_at=published,
        language=(markup.language if markup else None) or source.language,
    )


# --- fetching ------------------------------------------------------------------------------


class _BlockedError(Exception):
    """The host must not be fetched in this run."""


@dataclass
class _Host:
    robots: urllib.robotparser.RobotFileParser | None = None
    delay_s: float = 0.0
    last: float | None = None


class PageFetcher:
    """GETs pages on a publisher's own site, politely."""

    def __init__(self, http: httpx.AsyncClient, limits: IngestThresholds) -> None:
        self._http = http
        self._limits = limits
        self._hosts: dict[str, _Host] = {}
        self._blocked: set[str] = set()

    async def get(self, url: str, site: str) -> bytes | None:
        """The body of ``url``; None when it is not allowed, not found or too large."""
        try:
            for _ in range(MAX_REDIRECTS + 1):
                parts = urlsplit(url)
                host = (parts.hostname or "").lower()
                if parts.scheme not in {"http", "https"} or not host_on_site(host, site):
                    _log.info("fetch_refused", host=host, reason="off_site")
                    return None
                if not await self._allowed(parts.scheme, host, url, site):
                    _log.info("fetch_refused", host=host, reason="robots")
                    return None
                status, body, location = await self._request(host, url)
                if location is None:
                    return body if 200 <= status < 300 else None
                url = urljoin(url, location)
        except (_BlockedError, httpx.HTTPError, httpx.InvalidURL, TimeoutError) as exc:
            _log.info("fetch_failed", error=type(exc).__name__)
            return None
        _log.info("fetch_refused", reason="too_many_redirects")
        return None

    async def _pause(self, host: str) -> None:
        state = self._hosts.setdefault(host, _Host())
        if state.last is not None:
            wait = state.last + state.delay_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        state.last = time.monotonic()

    async def _request(self, host: str, url: str) -> tuple[int, bytes | None, str | None]:
        """Status, body (None when over the size cap) and redirect target."""
        await self._pause(host)
        limit = self._limits.max_page_bytes
        async with (
            asyncio.timeout(self._limits.fetch_timeout_s * 2),  # bounds a slow drip of bytes
            self._http.stream("GET", url, headers={"User-Agent": USER_AGENT}) as response,
        ):
            _log.debug("fetch", host=host, status=response.status_code)
            if response.is_redirect:
                return response.status_code, None, response.headers.get("location")
            if not response.is_success:
                return response.status_code, None, None
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > limit:
                    _log.info("fetch_refused", host=host, reason="too_large")
                    return response.status_code, None, None
                chunks.append(chunk)
        return response.status_code, b"".join(chunks), None

    async def _allowed(self, scheme: str, host: str, url: str, site: str) -> bool:
        if host in self._blocked:
            raise _BlockedError
        state = self._hosts.setdefault(host, _Host())
        if state.robots is None:
            state.robots = await self._load_robots(scheme, host, site)
            crawl_delay = state.robots.crawl_delay(USER_AGENT)
            state.delay_s = max(
                self._limits.per_host_delay_s,
                min(float(crawl_delay or 0), self._limits.max_crawl_delay_s),
            )
        return state.robots.can_fetch(USER_AGENT, url)

    async def _robots_response(self, scheme: str, host: str, site: str) -> tuple[int, bytes | None]:
        """robots.txt, following on-site redirects (http to https, bare host to www)."""
        url = f"{scheme}://{host}/robots.txt"
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            target = (parts.hostname or "").lower()
            if parts.scheme not in {"http", "https"} or not host_on_site(target, site):
                break
            status, body, location = await self._request(target, url)
            if location is None:
                return status, body
            url = urljoin(url, location)
        raise _BlockedError

    async def _load_robots(
        self, scheme: str, host: str, site: str
    ) -> urllib.robotparser.RobotFileParser:
        parser = urllib.robotparser.RobotFileParser()
        try:
            status, body = await self._robots_response(scheme, host, site)
        except (_BlockedError, httpx.HTTPError, httpx.InvalidURL, TimeoutError) as exc:
            self._blocked.add(host)
            raise _BlockedError from exc
        if 200 <= status < 300 and body is not None:
            parser.parse(body.decode("utf-8", errors="replace").splitlines())
        elif 400 <= status < 500:
            parser.parse([])  # no robots.txt: everything is allowed
        else:
            self._blocked.add(host)
            raise _BlockedError
        return parser


# --- ingest --------------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestDeps:
    catalogue: SourcesConfig
    limits: IngestThresholds
    index: FactcheckIndex
    embedder: Embedder
    fetcher: PageFetcher
    reviews: PublisherReviews | None


@dataclass(frozen=True)
class IngestOptions:
    lookback_days: int
    max_api_pages: int
    only: frozenset[str] | None = None


@dataclass
class SourceReport:
    source: str
    api_items: int = 0
    feed_items: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    upserted: int = 0
    error: str | None = None


@dataclass
class _Collected:
    records: dict[str, ArticleRecord] = field(default_factory=dict)

    def add(self, record: ArticleRecord) -> None:
        key = canonical_url(record.url)
        if len(key) > MAX_URL_CHARS:  # longer than the column; would fail the whole batch
            return
        old = self.records.get(key)
        if old is None:
            self.records[key] = record
            return
        self.records[key] = old.model_copy(
            update={
                "rating": old.rating or record.rating,
                "published_at": old.published_at or record.published_at,
                "language": old.language or record.language,
            }
        )

    def complete(self, url: str) -> bool:
        record = self.records.get(canonical_url(url))
        return record is not None and record.rating is not None and record.published_at is not None


async def _collect_api(
    deps: IngestDeps,
    options: IngestOptions,
    source: FactcheckSource,
    report: SourceReport,
    collected: _Collected,
) -> None:
    if deps.reviews is None or not source.api:
        return
    reviews = await deps.reviews.by_publisher(
        source.site,
        max_age_days=options.lookback_days,
        page_size=deps.limits.api_page_size,
        max_pages=options.max_api_pages,
    )
    for review in reviews:
        record = record_from_review(review, deps.catalogue)
        if record is not None:
            report.api_items += 1
            collected.add(record)


async def _new_entries(
    deps: IngestDeps,
    entries: list[FeedEntry],
    cutoff: dt.datetime,
    report: SourceReport,
    collected: _Collected,
) -> list[FeedEntry]:
    known = await deps.index.known_urls([entry.url for entry in entries])
    fresh: list[FeedEntry] = []
    seen: set[str] = set()
    for entry in entries:
        key = canonical_url(entry.url)
        too_old = entry.published_at is not None and entry.published_at < cutoff
        if too_old or key in known or key in seen or collected.complete(entry.url):
            report.skipped += 1
            continue
        seen.add(key)
        fresh.append(entry)
    return fresh


async def _collect_feeds(
    deps: IngestDeps,
    options: IngestOptions,
    source: FactcheckSource,
    report: SourceReport,
    collected: _Collected,
) -> None:
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=options.lookback_days)
    for feed in source.feeds:
        data = await deps.fetcher.get(feed, source.site)
        if data is None:
            report.failed += 1
            continue
        entries = await asyncio.to_thread(parse_feed, data, deps.limits.max_feed_items)
        report.feed_items += len(entries)
        for entry in await _new_entries(deps, entries, cutoff, report, collected):
            page = await deps.fetcher.get(entry.url, source.site)
            if page is None:
                report.failed += 1
                continue
            report.fetched += 1
            html = page.decode("utf-8", errors="replace")
            try:
                # HTML and date parsing is CPU-bound; keep it off the worker's event loop.
                record = await asyncio.to_thread(article_from_page, source, entry, html)
            except Exception as exc:  # one odd page must not drop the rest of the source
                _log.info("parse_failed", source=source.id, error=type(exc).__name__)
                report.failed += 1
                continue
            collected.add(record)


async def _store(deps: IngestDeps, records: Sequence[ArticleRecord]) -> int:
    size = deps.limits.embed_batch_size
    upserted = 0
    for start in range(0, len(records), size):
        batch = records[start : start + size]
        vectors = await deps.embedder.embed_passages([passage_text(r) for r in batch])
        upserted += await deps.index.upsert(
            [EmbeddedArticle(record, vector) for record, vector in zip(batch, vectors, strict=True)]
        )
    return upserted


async def run_ingest(deps: IngestDeps, options: IngestOptions) -> list[SourceReport]:
    """Index each source in turn; a failing source is reported and the next one still runs."""
    reports: list[SourceReport] = []
    for source in deps.catalogue.sources:
        if options.only is not None and source.id not in options.only:
            continue
        report = SourceReport(source=source.id)
        reports.append(report)
        collected = _Collected()
        try:
            await _collect_api(deps, options, source, report, collected)
        except Exception as exc:  # a failing search must not stop the feeds
            _log.warning("ingest_api_failed", source=source.id, exc_info=exc)
            report.error = type(exc).__name__
        try:
            await _collect_feeds(deps, options, source, report, collected)
            report.upserted = await _store(deps, list(collected.records.values()))
        except Exception as exc:  # one publisher failing must not stop the others
            _log.warning("ingest_source_failed", source=source.id, exc_info=exc)
            report.error = type(exc).__name__
        _log.info(
            "ingest_source",
            source=report.source,
            api_items=report.api_items,
            feed_items=report.feed_items,
            fetched=report.fetched,
            skipped=report.skipped,
            failed=report.failed,
            upserted=report.upserted,
            error=report.error,
        )
    return reports
