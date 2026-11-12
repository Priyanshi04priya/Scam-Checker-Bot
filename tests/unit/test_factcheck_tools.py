"""Tests for the Fact Check Tools client (mocked HTTP)."""

import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr

from scambot.mcp_server.clients.factcheck_tools import (
    SEARCH_URL,
    FactCheckToolsClient,
    FactCheckToolsError,
)

KEY = SecretStr("google-test-key")
FIXTURES = Path(__file__).parents[1] / "fixtures" / "factcheck"


def fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


async def test_search_sends_the_key_in_a_header_and_parses(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(SEARCH_URL).respond(200, json=fixture("claims_search.json"))

    async with httpx.AsyncClient() as http:
        reviews = await FactCheckToolsClient(http, KEY).search("free laptops", page_size=10)

    request = route.calls.last.request
    assert request.headers["X-Goog-Api-Key"] == "google-test-key"
    assert "google-test-key" not in str(request.url)
    assert dict(request.url.params) == {"query": "free laptops", "pageSize": "10"}
    assert [r.publisher_name for r in reviews] == ["Factly", "Vishvas News"]  # no-url one dropped
    first = reviews[0]
    assert first.claim == "Government is giving free laptops to students"
    assert first.claimant == "Social media users"
    assert (first.rating, first.language, first.publisher_site) == ("False", "en", "factly.in")
    assert first.review_date is not None
    assert first.review_date.utcoffset() is not None


async def test_empty_reply(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(SEARCH_URL).respond(200, json={})

    async with httpx.AsyncClient() as http:
        assert await FactCheckToolsClient(http, KEY).search("x", page_size=10) == []


async def test_publisher_paging_follows_tokens_up_to_max_pages(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.get(SEARCH_URL).mock(
        side_effect=[
            httpx.Response(200, json=fixture("claims_publisher_p1.json")),
            httpx.Response(200, json=fixture("claims_publisher_p2.json")),
        ]
    )

    async with httpx.AsyncClient() as http:
        reviews = await FactCheckToolsClient(http, KEY).by_publisher(
            "altnews.in", max_age_days=3, page_size=50, max_pages=2
        )

    assert len(reviews) == 2
    assert route.call_count == 2
    first, second = (dict(call.request.url.params) for call in route.calls)
    assert first == {"reviewPublisherSiteFilter": "altnews.in", "maxAgeDays": "3", "pageSize": "50"}
    assert second["pageToken"] == "page-2"
    assert "query" not in second


async def test_publisher_paging_stops_without_a_token(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(SEARCH_URL).respond(200, json=fixture("claims_search.json"))

    async with httpx.AsyncClient() as http:
        reviews = await FactCheckToolsClient(http, KEY).by_publisher(
            "factly.in", max_age_days=3, page_size=50, max_pages=5
        )

    assert len(reviews) == 2
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(500), "status_500"),
        (httpx.ConnectError("down"), "request_failed_ConnectError"),
        (httpx.Response(200, text="not json"), "unexpected_response"),
        (
            httpx.Response(
                200,
                json={
                    "claims": [{"text": "x", "claimReview": [{"url": "u", "reviewDate": "bad"}]}]
                },
            ),
            "unexpected_response",
        ),
    ],
)
async def test_failures_raise(
    respx_mock: respx.MockRouter, response: httpx.Response | Exception, code: str
) -> None:
    respx_mock.get(SEARCH_URL).mock(side_effect=[response])

    async with httpx.AsyncClient() as http:
        with pytest.raises(FactCheckToolsError) as raised:
            await FactCheckToolsClient(http, KEY).search("x", page_size=10)

    assert str(raised.value) == code
    assert "google-test-key" not in str(raised.value)
