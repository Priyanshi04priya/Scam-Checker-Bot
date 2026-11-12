"""Tests for fact-check ingest: parsing from fixtures, polite fetching (mocked HTTP) and runs."""

import asyncio
import datetime as dt
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from scambot.core.config import AppConfig, FactcheckSource, IngestThresholds, SourcesConfig
from scambot.factcheck_index.embed import SentenceTransformerEmbedder
from scambot.factcheck_index.ingest import (
    USER_AGENT,
    FeedEntry,
    IngestDeps,
    IngestOptions,
    PageFetcher,
    article_from_page,
    extract_claim_review,
    parse_feed,
    run_ingest,
)
from scambot.factcheck_index.records import (
    ArticleRecord,
    ClaimReview,
    canonical_url,
    normalise_language,
    record_from_review,
)
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from tests.fakes import FakePublisherReviews, KeywordEmbedder, StubEncoder, seed

FIXTURES = Path(__file__).parents[1] / "fixtures" / "factcheck"
SITE = "altnews.in"
BASE = "https://www.altnews.in"
NEW_URL = f"{BASE}/free-laptop-scheme-false/"
KNOWN_URL = f"{BASE}/old-flood-photo/"
OLD_URL = f"{BASE}/edited-minister-video/"
ALTNEWS = FactcheckSource(
    id="altnews", name="Alt News", site=SITE, language="en", feeds=[f"{BASE}/feed/"]
)


def read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def limits(app_config: AppConfig) -> IngestThresholds:
    return app_config.thresholds.ingest.model_copy(update={"per_host_delay_s": 0.0})


# --- parsing -------------------------------------------------------------------------------


def test_parse_feed() -> None:
    entries = parse_feed(read("altnews_feed.xml"), max_items=10)

    assert [e.url for e in entries] == [NEW_URL, KNOWN_URL, OLD_URL]
    assert entries[0].title == "Viral claim of free laptops under PM scheme is false"
    assert entries[1].published_at == dt.datetime(2026, 11, 9, 6, 30, tzinfo=dt.UTC)
    assert parse_feed(read("altnews_feed.xml"), max_items=1) == entries[:1]
    assert parse_feed(b"<rss><channel><item><title>no link</title></item></channel></rss>", 5) == []


def test_extract_claim_review_from_a_graph_with_a_bad_block() -> None:
    markup = extract_claim_review(read("article_claimreview.html").decode())

    assert markup is not None
    assert markup.claim is not None
    assert markup.claim.startswith("Government is giving free laptops")
    assert markup.rating == "False"
    assert markup.published_at == dt.datetime(2026, 11, 10, tzinfo=dt.UTC)
    assert (markup.language, markup.url) == ("en", NEW_URL)


def test_extract_claim_review_variants() -> None:
    page = (
        '<script type="application/ld+json">[{"@type": "ClaimReview", "claimReviewed": "x",'
        ' "reviewRating": {"name": "Misleading"}, "datePublished": "2026-11-01T10:00:00",'
        ' "inLanguage": {"alternateName": "hi"}}]</script>'
    )
    markup = extract_claim_review(page)

    assert markup is not None
    assert markup.rating == "Misleading"
    assert markup.published_at == dt.datetime(2026, 11, 1, 10, tzinfo=dt.UTC)
    assert markup.language == "hi"
    assert extract_claim_review(read("article_plain.html").decode()) is None
    assert (
        extract_claim_review('<script type="application/ld+json">{"@type": "News"}</script>')
        is None
    )


def test_article_from_page_with_claim_review() -> None:
    entry = FeedEntry(NEW_URL, "Feed title", dt.datetime(2026, 11, 10, 8, 30, tzinfo=dt.UTC))

    record = article_from_page(ALTNEWS, entry, read("article_claimreview.html").decode())

    assert record is not None
    assert record.claim.startswith("Government is giving free laptops")
    assert record.title == "Viral claim of free laptops under PM scheme is false"
    assert record.rating == "False"
    assert record.published_at == dt.datetime(2026, 11, 10, tzinfo=dt.UTC)
    assert (record.source, record.language) == ("altnews", "en")


def test_article_from_page_without_json_ld() -> None:
    hindi = ALTNEWS.model_copy(update={"language": "hi"})
    entry = FeedEntry(KNOWN_URL, "Feed title", None)

    record = article_from_page(hindi, entry, read("article_plain.html").decode())

    assert record is not None
    assert record.claim == "Feed title"
    assert record.title == "Old photo shared as recent flood"
    assert record.rating is None
    assert record.published_at is not None
    assert record.published_at.date() == dt.date(2026, 11, 9)
    assert record.language == "hi"


def test_canonical_url_and_language() -> None:
    assert (
        canonical_url("HTTPS://WWW.AltNews.in/Path/?utm_source=wa&id=3&fbclid=x&gclid=y#frag")
        == "https://www.altnews.in/Path/?id=3"
    )
    assert normalise_language("en-US") == "en"
    assert normalise_language({"alternateName": "hi"}) == "hi"
    assert normalise_language("English") is None
    assert normalise_language(7) is None


def test_record_from_review(app_config: AppConfig) -> None:
    catalogue = app_config.sources
    review = ClaimReview(claim=" A claim ", url="https://factly.in/x/", rating="False")
    record = record_from_review(review, catalogue)
    assert record is not None
    assert (record.source, record.title, record.claim) == ("factly", "A claim", "A claim")

    unknown = ClaimReview(claim="c", url="https://news.example.org/a", publisher_name="Desk")
    assert (record_from_review(unknown, catalogue) or record).source == "Desk"
    nameless = ClaimReview(claim="c", url="https://news.example.org/a")
    assert (record_from_review(nameless, catalogue) or record).source == "news.example.org"
    assert record_from_review(ClaimReview(claim="c", url="ftp://x/a"), catalogue) is None
    assert record_from_review(ClaimReview(claim=" ", url="https://x.in/a"), catalogue) is None


# --- fetching ------------------------------------------------------------------------------


async def test_robots_disallow_blocks_the_page(
    respx_mock: respx.MockRouter, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(200, text="User-agent: *\nDisallow: /private/")
    page = respx_mock.get(f"{BASE}/private/a").respond(200, text="secret")
    allowed = respx_mock.get(f"{BASE}/public/a").respond(200, text="hello")

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, limits)
        assert await fetcher.get(f"{BASE}/private/a", SITE) is None
        assert await fetcher.get(f"{BASE}/public/a", SITE) == b"hello"

    assert not page.called
    assert allowed.calls.last.request.headers["User-Agent"] == USER_AGENT


async def test_missing_robots_allows_everything(
    respx_mock: respx.MockRouter, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/a").respond(200, text="ok")
    respx_mock.get(f"{BASE}/gone").respond(410)

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, limits)
        assert await fetcher.get(f"{BASE}/a", SITE) == b"ok"
        assert await fetcher.get(f"{BASE}/gone", SITE) is None


@pytest.mark.parametrize("robots", [httpx.Response(503), httpx.ConnectError("down")])
async def test_unavailable_robots_skips_the_host(
    respx_mock: respx.MockRouter, limits: IngestThresholds, robots: httpx.Response | Exception
) -> None:
    robots_route = respx_mock.get(f"{BASE}/robots.txt").mock(side_effect=[robots])
    page = respx_mock.get(f"{BASE}/a").respond(200, text="ok")

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, limits)
        assert await fetcher.get(f"{BASE}/a", SITE) is None
        assert await fetcher.get(f"{BASE}/a", SITE) is None

    assert robots_route.call_count == 1
    assert not page.called


async def test_robots_redirect_on_site_is_followed(
    respx_mock: respx.MockRouter, limits: IngestThresholds
) -> None:
    bare = "https://altnews.in"
    respx_mock.get(f"{bare}/robots.txt").respond(301, headers={"location": f"{BASE}/robots.txt"})
    respx_mock.get(f"{BASE}/robots.txt").respond(200, text="User-agent: *\nDisallow: /private/")
    respx_mock.get(f"{bare}/a").respond(200, text="ok")
    private = respx_mock.get(f"{bare}/private/a").respond(200, text="secret")

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, limits)
        assert await fetcher.get(f"{bare}/a", SITE) == b"ok"
        assert await fetcher.get(f"{bare}/private/a", SITE) is None

    assert not private.called


async def test_off_site_urls_and_redirects_are_refused(
    respx_mock: respx.MockRouter, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/moved").respond(301, headers={"Location": "https://evil.example/x"})
    respx_mock.get(f"{BASE}/local").respond(302, headers={"Location": "/final"})
    respx_mock.get(f"{BASE}/final").respond(200, text="final")
    respx_mock.get(f"{BASE}/loop").respond(302, headers={"Location": "/loop"})
    evil = respx_mock.get("https://evil.example/x").respond(200)

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, limits)
        assert await fetcher.get("https://evil.example/x", SITE) is None
        assert await fetcher.get("https://notaltnews.in/x", SITE) is None
        assert await fetcher.get(f"{BASE}/moved", SITE) is None
        assert await fetcher.get(f"{BASE}/local", SITE) == b"final"
        assert await fetcher.get(f"{BASE}/loop", SITE) is None

    assert not evil.called


async def test_oversized_bodies_are_dropped_and_hosts_are_paced(
    respx_mock: respx.MockRouter, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/big").respond(200, content=b"x" * 50)
    respx_mock.get(f"{BASE}/small").respond(200, content=b"x" * 5)
    paced = limits.model_copy(update={"max_page_bytes": 10, "per_host_delay_s": 0.05})

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, paced)
        assert await fetcher.get(f"{BASE}/big", SITE) is None
        assert await fetcher.get(f"{BASE}/small", SITE) == b"xxxxx"


async def test_a_slow_drip_body_is_cut_off_by_the_deadline(
    respx_mock: respx.MockRouter, limits: IngestThresholds
) -> None:
    async def drip() -> AsyncIterator[bytes]:
        while True:
            yield b"x"
            await asyncio.sleep(0.01)

    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/slow").mock(return_value=httpx.Response(200, content=drip()))
    quick = limits.model_copy(update={"fetch_timeout_s": 0.05, "max_page_bytes": 10**9})

    async with httpx.AsyncClient() as http:
        fetcher = PageFetcher(http, quick)
        assert await fetcher.get(f"{BASE}/slow", SITE) is None


# --- runs ----------------------------------------------------------------------------------


def _lookback_days() -> int:
    """A lookback that keeps the November entries and drops the August one, on any date."""
    return (dt.datetime.now(dt.UTC) - dt.datetime(2026, 10, 1, tzinfo=dt.UTC)).days


def _catalogue(app_config: AppConfig, *sources: FactcheckSource) -> SourcesConfig:
    return SourcesConfig(sources=list(sources), ratings=app_config.sources.ratings)


async def test_run_ingest(
    respx_mock: respx.MockRouter, app_config: AppConfig, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/feed/").respond(200, content=read("altnews_feed.xml"))
    new_page = respx_mock.get(NEW_URL).respond(200, content=read("article_claimreview.html"))
    known_page = respx_mock.get(KNOWN_URL).respond(200)
    old_page = respx_mock.get(OLD_URL).respond(200)
    pib = FactcheckSource(id="pib", name="PIB Fact Check", site="pib.gov.in", language="en")
    api_review = ClaimReview(
        claim="A video shows floods in the city this week",
        url=f"{BASE}/old-flood-video/",
        rating="Misleading",
        review_date=dt.datetime(2026, 11, 8, tzinfo=dt.UTC),
    )
    reviews = FakePublisherReviews({"pib.gov.in": ConnectionError("down"), SITE: [api_review]})
    index = InMemoryFactcheckIndex()
    await seed(
        index,
        KeywordEmbedder(),
        ArticleRecord(source="altnews", url=KNOWN_URL, title="t", claim="c"),
    )
    encoder = StubEncoder()

    async with httpx.AsyncClient() as http:
        deps = IngestDeps(
            catalogue=_catalogue(app_config, pib, ALTNEWS),
            limits=limits,
            index=index,
            embedder=SentenceTransformerEmbedder("e5", loader=lambda _name: encoder),
            fetcher=PageFetcher(http, limits),
            reviews=reviews,
        )
        failed, altnews = await run_ingest(
            deps, IngestOptions(lookback_days=_lookback_days(), max_api_pages=2)
        )

    assert (failed.source, failed.error) == ("pib", "ConnectionError")
    assert altnews.error is None
    assert (altnews.api_items, altnews.feed_items, altnews.fetched) == (1, 3, 1)
    assert (altnews.skipped, altnews.failed, altnews.upserted) == (2, 0, 2)
    assert new_page.call_count == 1
    assert not known_page.called
    assert not old_page.called
    assert reviews.calls[1] == (SITE, _lookback_days(), limits.api_page_size, 2)
    assert set(index.items) == {KNOWN_URL, NEW_URL, f"{BASE}/old-flood-video/"}
    assert index.items[NEW_URL].record.rating == "False"
    ((texts, _kwargs),) = encoder.calls
    assert all(text.startswith("passage: ") for text in texts)


async def test_run_ingest_without_the_api_and_only_selected_sources(
    respx_mock: respx.MockRouter, app_config: AppConfig, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/feed/").respond(200, content=read("altnews_feed.xml"))
    respx_mock.get(NEW_URL).respond(200, content=read("article_claimreview.html"))
    respx_mock.get(KNOWN_URL).respond(500)
    broken_feed = ALTNEWS.model_copy(update={"id": "altnews_hi", "feeds": [f"{BASE}/hindi/feed/"]})
    respx_mock.get(f"{BASE}/hindi/feed/").respond(404)
    other = FactcheckSource(id="boom", name="BOOM", site="boomlive.in", language="en")
    index = InMemoryFactcheckIndex()

    async with httpx.AsyncClient() as http:
        deps = IngestDeps(
            catalogue=_catalogue(app_config, ALTNEWS, broken_feed, other),
            limits=limits,
            index=index,
            embedder=KeywordEmbedder(),
            fetcher=PageFetcher(http, limits),
            reviews=None,
        )
        reports = await run_ingest(
            deps,
            IngestOptions(
                lookback_days=_lookback_days(),
                max_api_pages=1,
                only=frozenset({"altnews", "altnews_hi"}),
            ),
        )

    assert [r.source for r in reports] == ["altnews", "altnews_hi"]
    assert (reports[0].upserted, reports[0].failed) == (1, 1)
    assert (reports[1].failed, reports[1].upserted) == (1, 0)
    assert set(index.items) == {NEW_URL}


async def test_failing_api_still_indexes_the_feeds(
    respx_mock: respx.MockRouter, app_config: AppConfig, limits: IngestThresholds
) -> None:
    respx_mock.get(f"{BASE}/robots.txt").respond(404)
    respx_mock.get(f"{BASE}/feed/").respond(200, content=read("altnews_feed.xml"))
    respx_mock.get(NEW_URL).respond(200, content=read("article_claimreview.html"))
    respx_mock.get(KNOWN_URL).respond(404)
    index = InMemoryFactcheckIndex()

    async with httpx.AsyncClient() as http:
        deps = IngestDeps(
            catalogue=_catalogue(app_config, ALTNEWS),
            limits=limits,
            index=index,
            embedder=KeywordEmbedder(),
            fetcher=PageFetcher(http, limits),
            reviews=FakePublisherReviews({SITE: ConnectionError("down")}),
        )
        [report] = await run_ingest(
            deps, IngestOptions(lookback_days=_lookback_days(), max_api_pages=1)
        )

    assert report.error == "ConnectionError"
    assert report.upserted == 1
    assert set(index.items) == {NEW_URL}
