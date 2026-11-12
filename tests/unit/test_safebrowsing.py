"""Tests for the Safe Browsing client (mocked HTTP)."""

import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from scambot.mcp_server.clients.safebrowsing import FIND_URL, SafeBrowsingClient, SafeBrowsingError

KEY = SecretStr("google-test-key")


async def test_key_is_sent_in_a_header_and_matches_parsed(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(FIND_URL).respond(
        200,
        json={
            "matches": [
                {"threat": {"url": "http://bad.example/"}, "threatType": "SOCIAL_ENGINEERING"}
            ]
        },
    )

    async with httpx.AsyncClient() as http:
        unsafe = await SafeBrowsingClient(http, KEY).unsafe_urls(
            ["http://bad.example/", "https://ok.example/"]
        )

    assert unsafe == {"http://bad.example/": "SOCIAL_ENGINEERING"}
    request = route.calls.last.request
    assert request.headers["X-Goog-Api-Key"] == "google-test-key"
    assert "google-test-key" not in str(request.url)
    body = json.loads(request.content)
    assert body["threatInfo"]["threatEntries"] == [
        {"url": "http://bad.example/"},
        {"url": "https://ok.example/"},
    ]


async def test_empty_response_means_no_matches(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(FIND_URL).respond(200, json={})

    async with httpx.AsyncClient() as http:
        assert await SafeBrowsingClient(http, KEY).unsafe_urls(["https://ok.example/"]) == {}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"matches": [{"threat": {}}]}),
    ],
)
async def test_failures_raise(respx_mock: respx.MockRouter, response: httpx.Response) -> None:
    respx_mock.post(FIND_URL).mock(return_value=response)

    async with httpx.AsyncClient() as http:
        with pytest.raises(SafeBrowsingError):
            await SafeBrowsingClient(http, KEY).unsafe_urls(["https://x.example/"])


async def test_network_error_raises(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(FIND_URL).mock(side_effect=httpx.ConnectTimeout("slow"))

    async with httpx.AsyncClient() as http:
        with pytest.raises(SafeBrowsingError):
            await SafeBrowsingClient(http, KEY).unsafe_urls(["https://x.example/"])


async def test_empty_input_makes_no_request(respx_mock: respx.MockRouter) -> None:
    async with httpx.AsyncClient() as http:
        assert await SafeBrowsingClient(http, KEY).unsafe_urls([]) == {}

    assert not respx_mock.calls
