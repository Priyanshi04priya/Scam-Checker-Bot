"""Tests for reading a news page's title, description and date."""

import datetime as dt
from pathlib import Path

import pytest

from scambot.mcp_server.page_extract import PageInfo, extract_page, parse_date

PAGES = Path(__file__).parent.parent / "fixtures" / "pages"
TODAY = dt.date(2026, 11, 13)
URL = "https://www.thehindu.com/news/story"


def read(name: str) -> str:
    return (PAGES / f"{name}.html").read_text(encoding="utf-8")


def info(name: str) -> PageInfo:
    return extract_page(read(name), URL, min_year=1995, today=TODAY)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("jsonld_graph", dt.date(2019, 8, 10)),
        ("jsonld_list", dt.date(2018, 3, 4)),
        ("og_offset", dt.date(2020, 1, 15)),  # the date as written, no timezone shift
        ("meta_name_date", dt.date(2017, 6, 1)),
        ("itemprop", dt.date(2016, 2, 29)),
        ("time_article", dt.date(2015, 12, 25)),
        ("time_pubdate", dt.date(2014, 7, 4)),
        ("malformed_then_valid", dt.date(2012, 11, 10)),
        ("no_date", None),
        ("future_date", None),
        ("ancient_date", None),
    ],
)
def test_publication_date(name: str, expected: dt.date | None) -> None:
    assert info(name).published == expected


def test_trafilatura_is_the_last_resort() -> None:
    assert info("trafilatura_only").published in (None, dt.date(2013, 5, 6))


def test_title_description_and_page_claimed_publisher() -> None:
    graph = info("jsonld_graph")
    assert graph.title == "Flood hits Kochi"
    assert graph.description == "Heavy rain flooded Kochi streets."
    assert graph.publisher == "Some Page Claim"
    og = info("og_offset")
    assert (og.title, og.description, og.publisher) == (
        "Og title here",
        "Og description here",
        "Their Name",
    )


def test_title_falls_back_to_the_title_tag() -> None:
    assert info("meta_name_date").title == "T"


def test_unparseable_markup_gives_nothing() -> None:
    result = extract_page("", URL, min_year=1995, today=TODAY)
    assert (result.title, result.published) == (None, None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019-08-10T09:30:00Z", dt.date(2019, 8, 10)),
        ("2019-08-10", dt.date(2019, 8, 10)),
        ("2019-08-10T09:30:00+05:30", dt.date(2019, 8, 10)),
        ("20190810", dt.date(2019, 8, 10)),
        ("2019", None),
        ("2019-08", None),
        ("garbage", None),
        ("", None),
        (None, None),
        (5, None),
    ],
)
def test_parse_date(value: object, expected: dt.date | None) -> None:
    assert parse_date(value) == expected
