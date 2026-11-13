"""Tests for the provenance checker."""

import datetime as dt
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scambot.core.config import AppConfig
from scambot.core.contracts import CheckerResult
from scambot.core.media_store import ORIGINAL, MediaStore
from scambot.mcp_server.checkers.provenance import (
    ProvenanceChecker,
    read_c2pa,
    read_exif,
    utc_today,
)
from scambot.mcp_server.clients.pagefetch import FetchError
from scambot.mcp_server.clients.vision import VisionError, WebPage
from scambot.mcp_server.media import MediaFiles
from tests.fakes import FakePageFetcher, FakeWebSearch, StaticClock
from tests.media import exif_jpeg, png_bytes, store_bytes

PAGES = Path(__file__).parent.parent / "fixtures" / "pages"
KEY = "job1"
TODAY = dt.date(2026, 11, 13)
HINDU = "https://www.thehindu.com/news/kochi-floods"
NDTV = "https://www.ndtv.com/photos/old"
BLOG = "https://random-blog.example/flood"


def page_html(published: str | None) -> str:
    meta = f'<meta property="article:published_time" content="{published}">' if published else ""
    return f"<html><head><title>Story {published}</title>{meta}</head><body>x</body></html>"


async def store_image(store: MediaStore, data: bytes | None = None) -> None:
    await store_bytes(store, KEY, ORIGINAL, data or exif_jpeg())


def build(
    app_config: AppConfig,
    store: MediaStore,
    search: FakeWebSearch | None,
    fetcher: FakePageFetcher,
) -> ProvenanceChecker:
    return ProvenanceChecker(
        search,
        fetcher,
        MediaFiles(store),
        app_config.thresholds.provenance,
        app_config.sources,
        today=StaticClock(TODAY),
    )


async def run(
    app_config: AppConfig,
    store: MediaStore,
    pages: list[WebPage] | Exception,
    html: dict[str, str | FetchError],
    reference: dt.date | None = None,
) -> tuple[CheckerResult, FakePageFetcher]:
    await store_image(store)
    fetcher = FakePageFetcher(html)
    checker = build(app_config, store, FakeWebSearch(pages), fetcher)
    return await checker.check(KEY, reference), fetcher


async def test_old_picture_against_a_recent_claim(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    result, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, "Kochi", "full")],
        {HINDU: page_html("2019-08-10T09:30:00+05:30")},
        dt.date(2024, 5, 1),
    )

    assert (result.status, result.signal, result.confidence) == ("ok", "old_media", 0.85)
    origin = result.details["origin"]
    assert origin["url"] == HINDU
    assert origin["publisher"] == "The Hindu"
    assert origin["domain"] == "www.thehindu.com"
    assert origin["credible"] is True
    assert origin["date"] == "2019-08-10"
    assert origin["age_days"] == (dt.date(2024, 5, 1) - dt.date(2019, 8, 10)).days
    assert result.evidence[0].url == HINDU
    assert result.evidence[0].date == dt.date(2019, 8, 10)
    assert result.details["exif"]["captured_at"] == "2019-08-10"
    assert result.details["exif"]["device"] == "Acme Cam 1"


async def test_reference_defaults_to_today(app_config: AppConfig, media_store: MediaStore) -> None:
    result, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, None, "partial")],
        {HINDU: page_html("2019-08-10")},
    )

    assert (result.signal, result.confidence) == ("old_media", 0.70)
    assert result.details["reference_date"] == TODAY.isoformat()


@pytest.mark.parametrize(("days", "signal"), [(30, "none"), (31, "old_media"), (-5, "none")])
async def test_age_boundary(
    app_config: AppConfig, media_store: MediaStore, days: int, signal: str
) -> None:
    published = TODAY - dt.timedelta(days=days)
    result, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, None, "full")],
        {HINDU: page_html(published.isoformat())},
    )

    assert (result.status, result.signal) == ("ok", signal)


async def test_recent_credible_page_confidence(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    full, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, None, "full")],
        {HINDU: page_html(TODAY.isoformat())},
    )
    partial, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, None, "partial")],
        {HINDU: page_html(TODAY.isoformat())},
    )

    assert (full.signal, full.confidence) == ("none", 0.80)
    assert (partial.signal, partial.confidence) == ("none", 0.65)


async def test_earliest_credible_page_wins_and_others_are_listed(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    result, fetcher = await run(
        app_config,
        media_store,
        [
            WebPage(HINDU, None, "full"),
            WebPage(BLOG, None, "full"),
            WebPage(NDTV, None, "partial"),
        ],
        {
            HINDU: page_html("2021-01-01"),
            NDTV: page_html("2019-03-03"),
            BLOG: page_html("2001-01-01"),
        },
        dt.date(2024, 1, 1),
    )

    assert result.details["origin"]["url"] == NDTV
    assert [e.url for e in result.evidence] == [NDTV, HINDU]
    assert BLOG not in fetcher.calls  # unlisted sites are never fetched
    assert result.details["candidates"] == 2


async def test_no_match_and_no_credible_page(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    none, _ = await run(app_config, media_store, [], {})
    blog, fetcher = await run(app_config, media_store, [WebPage(BLOG, None, "full")], {})

    assert (none.status, none.details["reason"]) == ("inconclusive", "no_match")
    assert (blog.status, blog.details["reason"]) == ("inconclusive", "no_credible_page")
    assert fetcher.calls == []
    assert "origin" not in blog.details


async def test_credible_but_undated(app_config: AppConfig, media_store: MediaStore) -> None:
    result, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, None, "full"), WebPage(NDTV, None, "full")],
        {HINDU: page_html(None), NDTV: FetchError("status_404")},
    )

    assert (result.status, result.details["reason"]) == ("inconclusive", "no_dated_page")
    assert [p["error"] for p in result.details["pages"]] == [None, "status_404"]


async def test_fetch_failures_drop_only_that_page(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    result, _ = await run(
        app_config,
        media_store,
        [WebPage(HINDU, None, "full"), WebPage(NDTV, None, "full")],
        {HINDU: FetchError("timeout"), NDTV: page_html("2020-02-02")},
        dt.date(2024, 1, 1),
    )

    assert result.details["origin"]["url"] == NDTV


async def test_search_failures_are_errors_with_embedded_details_kept(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    failed, _ = await run(app_config, media_store, VisionError("status_403"), {})

    assert (failed.status, failed.error) == ("error", "status_403")
    assert failed.details["exif"]["captured_at"] == "2019-08-10"


async def test_search_timeout(
    app_config: AppConfig, media_store: MediaStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await store_image(media_store)
    limits = app_config.thresholds.provenance.model_copy(update={"vision_timeout_s": 0.01})

    class Slow:
        async def pages_with_matches(self, image: bytes, *, max_results: int) -> list[WebPage]:
            import asyncio  # noqa: PLC0415

            await asyncio.sleep(1)
            return []

    checker = ProvenanceChecker(
        Slow(), FakePageFetcher(), MediaFiles(media_store), limits, app_config.sources
    )

    result = await checker.check(KEY, None)

    assert (result.status, result.error) == ("error", "timeout")


async def test_not_configured_and_missing_media(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    await store_image(media_store)
    unconfigured = build(app_config, media_store, None, FakePageFetcher())
    missing = build(app_config, media_store, FakeWebSearch(), FakePageFetcher())

    no_key = await unconfigured.check(KEY, None)
    gone = await missing.check("nothere", None)

    assert (no_key.status, no_key.error) == ("error", "not_configured")
    assert (gone.status, gone.error) == ("error", "media_missing")


async def test_undecodable_image(app_config: AppConfig, media_store: MediaStore) -> None:
    await store_image(media_store, b"not an image at all")
    search = FakeWebSearch()
    checker = build(app_config, media_store, search, FakePageFetcher())

    result = await checker.check(KEY, None)

    assert (result.status, result.error) == ("error", "unreadable_image")
    assert search.calls == []


async def test_large_image_is_shrunk_before_search(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    await store_image(media_store, png_bytes((3000, 2000)))
    search = FakeWebSearch()

    await build(app_config, media_store, search, FakePageFetcher()).check(KEY, None)

    assert len(search.calls) == 1
    assert search.calls[0][1] == app_config.thresholds.provenance.max_results


def test_read_exif() -> None:
    info = read_exif(exif_jpeg("2018:01:02 03:04:05"))
    assert (info.captured_at, info.device, info.software) == (
        dt.date(2018, 1, 2),
        "Acme Cam 1",
        "EditorX",
    )
    assert read_exif(exif_jpeg("garbage")).captured_at is None
    assert read_exif(exif_jpeg("2018:13:45 00:00:00")).captured_at is None
    assert read_exif(b"junk").device is None
    assert read_exif(png_bytes()) == read_exif(b"junk")


def test_utc_today() -> None:
    assert abs((utc_today() - dt.datetime.now(dt.UTC).date()).days) <= 1


def test_c2pa_unavailable_when_the_library_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "c2pa", None)

    assert read_c2pa(b"x", "image/jpeg") == {"status": "unavailable"}


def fake_c2pa(reader: Any) -> types.ModuleType:
    module = types.ModuleType("c2pa")
    module.Reader = reader  # type: ignore[attr-defined]
    return module


def test_c2pa_absent_failed_and_present(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_mime: str, _stream: object) -> None:
        msg = "ManifestNotFound: no manifest"
        raise RuntimeError(msg)

    def broken(_mime: str, _stream: object) -> None:
        msg = "Other trouble"
        raise RuntimeError(msg)

    class Present:
        def __init__(self, _mime: str, _stream: object) -> None: ...

        def json(self) -> str:
            return (
                '{"active_manifest":"m1","validation_state":"Valid","manifests":{"m1":'
                '{"signature_info":{"issuer":"Acme CA"},"assertions":[{"label":"c2pa.actions.v2",'
                '"data":{"actions":[{"action":"c2pa.created"},{"action":"c2pa.edited"}]}}]}}}'
            )

    class Garbled:
        def __init__(self, _mime: str, _stream: object) -> None: ...

        def json(self) -> str:
            return "[1, 2]"

    monkeypatch.setitem(sys.modules, "c2pa", fake_c2pa(missing))
    assert read_c2pa(b"x", "image/jpeg") == {"status": "absent"}
    monkeypatch.setitem(sys.modules, "c2pa", fake_c2pa(broken))
    assert read_c2pa(b"x", "image/jpeg") == {"status": "failed"}
    monkeypatch.setitem(sys.modules, "c2pa", fake_c2pa(Present))
    assert read_c2pa(b"x", "image/jpeg") == {
        "status": "present",
        "valid": True,
        "issuer": "Acme CA",
        "actions": ["c2pa.created", "c2pa.edited"],
    }
    monkeypatch.setitem(sys.modules, "c2pa", fake_c2pa(Garbled))
    assert read_c2pa(b"x", "image/jpeg") == {"status": "failed"}


def test_c2pa_real_library_reports_absent_for_a_plain_image() -> None:
    pytest.importorskip("c2pa")

    assert read_c2pa(exif_jpeg(), "image/jpeg")["status"] in ("absent", "failed")


async def test_a_named_picture_is_searched_instead_of_the_original(
    app_config: AppConfig, media_store: MediaStore
) -> None:
    await store_bytes(media_store, KEY, ORIGINAL, b"not a picture, e.g. a video")
    await store_bytes(media_store, KEY, "kf_00.jpg", exif_jpeg())
    checker = build(
        app_config,
        media_store,
        FakeWebSearch([WebPage(HINDU, None, "full")]),
        FakePageFetcher({HINDU: page_html("2020-01-01")}),
    )

    result = await checker.check(KEY, TODAY, "kf_00.jpg")

    assert (result.status, result.signal) == ("ok", "old_media")
