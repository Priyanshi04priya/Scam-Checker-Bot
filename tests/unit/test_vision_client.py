"""Tests for the Vision web detection client."""

import base64
import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from scambot.mcp_server.clients.vision import ANNOTATE_URL, VisionClient, VisionError, WebPage


def client(http: httpx.AsyncClient) -> VisionClient:
    return VisionClient(http, SecretStr("g-key"))


def detection(*pages: dict[str, Any]) -> dict[str, Any]:
    return {"responses": [{"webDetection": {"pagesWithMatchingImages": list(pages)}}]}


async def test_request_shape_and_page_mapping() -> None:
    body = detection(
        {"url": "https://a.example/1", "pageTitle": "One", "partialMatchingImages": [{"url": "x"}]},
        {"url": "https://a.example/2", "fullMatchingImages": [{"url": "y"}]},
        {"url": "https://a.example/1", "fullMatchingImages": [{"url": "z"}]},
        {"pageTitle": "no url"},
        {"url": "https://a.example/2", "partialMatchingImages": [{"url": "y"}]},
    )
    async with respx.mock() as router, httpx.AsyncClient() as http:
        route = router.post(ANNOTATE_URL).respond(json=body)

        pages = await client(http).pages_with_matches(b"img", max_results=7)

    request = route.calls.last.request
    assert request.headers["X-Goog-Api-Key"] == "g-key"
    assert "g-key" not in str(request.url)
    sent = json.loads(request.content)["requests"][0]
    assert base64.b64decode(sent["image"]["content"]) == b"img"
    assert sent["features"] == [{"type": "WEB_DETECTION", "maxResults": 7}]
    assert pages == [
        WebPage("https://a.example/1", "One", "full"),
        WebPage("https://a.example/2", None, "full"),
    ]


async def test_no_web_detection_is_an_empty_list() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(ANNOTATE_URL).respond(json={"responses": [{}]})

        assert await client(http).pages_with_matches(b"i", max_results=1) == []


@pytest.mark.parametrize(
    ("respond", "code"),
    [
        ({"json": {"responses": [{"error": {"code": 3, "message": "bad"}}]}}, "vision_error_3"),
        ({"status_code": 403}, "status_403"),
        ({"status_code": 500}, "status_500"),
        ({"content": b"not json"}, "unexpected_response"),
        ({"json": {"responses": []}}, "unexpected_response"),
    ],
)
async def test_failures_have_short_codes(respond: dict[str, Any], code: str) -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(ANNOTATE_URL).respond(**respond)

        with pytest.raises(VisionError, match=code):
            await client(http).pages_with_matches(b"i", max_results=1)


async def test_timeout_has_a_short_code() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(ANNOTATE_URL).mock(side_effect=httpx.ReadTimeout("slow"))

        with pytest.raises(VisionError, match="request_failed_ReadTimeout"):
            await client(http).pages_with_matches(b"i", max_results=1)
