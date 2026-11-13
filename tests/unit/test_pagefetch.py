"""Tests for the SSRF-safe page fetcher."""

import asyncio
from typing import Any

import httpx
import pytest
import respx

from scambot.core.config import FetchThresholds
from scambot.mcp_server.clients.pagefetch import FetchError, SafePageFetcher

LIMITS = FetchThresholds(timeout_s=2, max_bytes=1000, max_redirects=2)
PUBLIC = "93.184.216.34"
HTML = {"content-type": "text/html; charset=utf-8"}


def resolver(table: dict[str, list[str]] | None = None) -> Any:
    answers = table or {}

    async def resolve(host: str) -> list[str]:
        if host not in answers and "*" not in answers:
            raise OSError(host)
        return answers.get(host, answers.get("*", []))

    return resolve


def fetcher(http: httpx.AsyncClient, table: dict[str, list[str]] | None = None) -> SafePageFetcher:
    return SafePageFetcher(http, LIMITS, resolver=resolver(table or {"*": [PUBLIC]}))


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("file:///etc/passwd", "scheme"),
        ("ftp://news.example/x", "scheme"),
        ("http://127.0.0.1/", "host_not_public"),
        ("http://localhost/", "host_not_public"),
        ("http://intranet/", "host_not_public"),
        ("http://box.local/", "host_not_public"),
        ("http://svc.internal/", "host_not_public"),
        ("http://10.0.0.5/", "host_not_public"),
        ("http://169.254.169.254/latest", "host_not_public"),
        ("http://[::1]/", "host_not_public"),
        ("http://[::ffff:127.0.0.1]/", "host_not_public"),
        ("http://100.64.0.1/", "host_not_public"),
        ("http://[2002:7f00:1::]/", "host_not_public"),
        ("http://[2002:a9fe:a9fe::1]/", "host_not_public"),
        ("http://[64:ff9b::a00:1]/", "host_not_public"),
        ("http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/", "host_not_public"),
        ("http://news.example:8080/", "port"),
        ("http://news.example:abc/", "port"),
        ("http://user:pw@news.example/", "userinfo"),
        ("http:///x", "host_not_public"),
    ],
)
async def test_unsafe_urls_are_refused_before_any_request(url: str, code: str) -> None:
    async with respx.mock(assert_all_called=False) as router, httpx.AsyncClient() as http:
        route = router.route().respond(200)

        with pytest.raises(FetchError, match=code):
            await fetcher(http).get_html(url)

    assert route.call_count == 0


async def test_one_private_answer_is_enough_to_refuse() -> None:
    async with httpx.AsyncClient() as http:
        mixed = fetcher(http, {"news.example": [PUBLIC, "10.0.0.1"]})

        with pytest.raises(FetchError, match="host_not_public"):
            await mixed.get_html("https://news.example/")


async def test_dns_failure() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(FetchError, match="dns_failed"):
            await fetcher(http, {"other.example": [PUBLIC]}).get_html("https://news.example/")
        with pytest.raises(FetchError, match="dns_failed"):
            await fetcher(http, {"news.example": []}).get_html("https://news.example/")


async def test_connects_to_the_checked_address() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        route = router.get(f"https://{PUBLIC}/story").respond(200, text="<p>hi</p>", headers=HTML)

        html = await fetcher(http).get_html("https://news.example/story")

    assert html == "<p>hi</p>"
    request = route.calls.last.request
    assert request.headers["Host"] == "news.example"
    assert request.extensions["sni_hostname"] == "news.example"
    assert request.headers["User-Agent"].startswith("scambot/")


async def test_ipv6_address_is_bracketed() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        route = router.get("https://[2606:2800:220:1::1]/").respond(200, text="x", headers=HTML)

        await fetcher(http, {"*": ["2606:2800:220:1::1"]}).get_html("https://news.example/")

    assert route.called


async def test_redirect_to_a_private_host_is_refused() -> None:
    table = {"news.example": [PUBLIC], "evil.example": ["10.0.0.9"]}
    async with respx.mock(assert_all_called=False) as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/a").respond(
            302, headers={"location": "https://evil.example/"}
        )
        private = router.get("https://10.0.0.9/").respond(200, text="secret", headers=HTML)

        with pytest.raises(FetchError, match="host_not_public"):
            await fetcher(http, table).get_html("https://news.example/a")

    assert private.call_count == 0


async def test_relative_redirect_is_followed() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/a").respond(301, headers={"location": "/b"})
        router.get(f"https://{PUBLIC}/b").respond(200, text="ok", headers=HTML)

        assert await fetcher(http).get_html("https://news.example/a") == "ok"


async def test_redirect_loop_and_missing_location() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/loop").respond(302, headers={"location": "/loop"})
        router.get(f"https://{PUBLIC}/bare").respond(302)

        with pytest.raises(FetchError, match="redirect_limit"):
            await fetcher(http).get_html("https://news.example/loop")
        with pytest.raises(FetchError, match="request_failed"):
            await fetcher(http).get_html("https://news.example/bare")


async def test_oversize_body() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/big").respond(200, content=b"a" * 5000, headers=HTML)

        with pytest.raises(FetchError, match="too_large"):
            await fetcher(http).get_html("https://news.example/big")


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"status_code": 404}, "status_404"),
        (
            {"status_code": 200, "content": b"x", "headers": {"content-type": "image/png"}},
            "not_html",
        ),
        ({"status_code": 200, "content": b"x"}, "not_html"),
    ],
)
async def test_bad_responses(kwargs: dict[str, Any], code: str) -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/p").respond(**kwargs)

        with pytest.raises(FetchError, match=code):
            await fetcher(http).get_html("https://news.example/p")


async def test_slow_server_times_out() -> None:
    async def slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200)

    async with respx.mock(assert_all_called=False) as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/slow").mock(side_effect=slow)
        short = SafePageFetcher(
            http,
            FetchThresholds(timeout_s=0.05, max_bytes=1000, max_redirects=1),
            resolver=resolver({"*": [PUBLIC]}),
        )

        with pytest.raises(FetchError, match="timeout"):
            await short.get_html("https://news.example/slow")


async def test_transport_errors_are_request_failed() -> None:
    async with respx.mock() as router, httpx.AsyncClient() as http:
        router.get(f"https://{PUBLIC}/x").mock(side_effect=httpx.ConnectError("down"))
        router.get(f"https://{PUBLIC}/t").mock(side_effect=httpx.ReadTimeout("slow"))

        with pytest.raises(FetchError, match="request_failed"):
            await fetcher(http).get_html("https://news.example/x")
        with pytest.raises(FetchError, match="timeout"):
            await fetcher(http).get_html("https://news.example/t")


async def test_system_resolver_resolves_a_literal() -> None:
    from scambot.mcp_server.clients.pagefetch import system_resolver  # noqa: PLC0415

    assert await system_resolver("127.0.0.1") == ["127.0.0.1"]
