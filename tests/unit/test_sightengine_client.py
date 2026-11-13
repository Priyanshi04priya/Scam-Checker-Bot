"""Tests for the Sightengine client."""

from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr

from scambot.mcp_server.clients.sightengine import (
    CHECK_URL,
    SightengineClient,
    SightengineError,
    SynthScores,
)


def client(http: httpx.AsyncClient) -> SightengineClient:
    return SightengineClient(http, "the-user", SecretStr("the-secret"))


async def test_request_and_scores() -> None:
    body = {"status": "success", "type": {"ai_generated": 0.93, "deepfake": 1.4}}
    async with respx.mock() as router, httpx.AsyncClient() as http:
        route = router.post(CHECK_URL).respond(json=body)

        scores = await client(http).scores(b"pixels", "original")

    request = route.calls.last.request
    assert "the-secret" not in str(request.url)
    assert request.url.query == b""
    content = request.content
    for part in (b'name="media"', b"pixels", b"genai,deepfake", b"the-user", b"the-secret"):
        assert part in content
    assert scores == SynthScores(0.93, 1.0)


async def test_missing_scores_are_none() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(CHECK_URL).respond(json={"status": "success", "type": {"ai_generated": 0.2}})

        assert await client(http).scores(b"p", "f") == SynthScores(0.2, None)


async def test_odd_values_are_none() -> None:
    body = {"status": "success", "type": {"ai_generated": "high", "deepfake": True}}
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(CHECK_URL).respond(json=body)

        assert await client(http).scores(b"p", "f") == SynthScores(None, None)


@pytest.mark.parametrize(
    ("respond", "code"),
    [
        ({"json": {"status": "failure", "error": {"code": 21}}}, "vendor_error_21"),
        ({"json": {"status": "failure"}}, "vendor_failure"),
        ({"status_code": 401}, "status_401"),
        ({"content": b"nope"}, "unexpected_response"),
    ],
)
async def test_failures(respond: dict[str, Any], code: str) -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(CHECK_URL).respond(**respond)

        with pytest.raises(SightengineError, match=code):
            await client(http).scores(b"p", "f")


async def test_network_failure() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.post(CHECK_URL).mock(side_effect=httpx.ConnectError("down"))

        with pytest.raises(SightengineError, match="request_failed_ConnectError"):
            await client(http).scores(b"p", "f")
