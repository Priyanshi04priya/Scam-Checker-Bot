"""Tests for entity extraction and short-link expansion."""

import asyncio

import httpx
import pytest
import respx

from scambot.core.contracts import Entities, UrlEntity
from scambot.preprocess.entities import (
    EntityLimits,
    HeadRedirectExpander,
    NoopExpander,
    extract_entities,
    find_entities,
)

SHORTENERS = frozenset({"bit.ly", "tinyurl.com"})
LIMITS = EntityLimits(max_per_kind=20, expand_timeout_s=1.0)


def urls(text: str) -> list[str]:
    return [entity.url for entity in find_entities(text, 20).urls]


@pytest.mark.parametrize(
    "text",
    [
        "see file.txt for details",
        "e.g. this one",
        "Mr.Sharma called",
        "pi is 3.14 roughly",
        "reference 123456789012 noted",
        "version 1.2.3 released",
    ],
)
def test_ordinary_text_has_no_urls(text: str) -> None:
    assert find_entities(text, 20) == Entities()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("open bit.ly/3xYz now", ["bit.ly/3xYz"]),
        ("win at amazon-offer.xyz/win!", ["amazon-offer.xyz/win"]),
        ("apply on pmkisan-yojana.online today", ["pmkisan-yojana.online"]),
        ("visit https://www.sbi.co.in/web/kyc).", ["https://www.sbi.co.in/web/kyc"]),
        ("www.paytrn.com पर जाएं।", ["www.paytrn.com"]),
        ("दो लिंक: bit.ly/a और bit.ly/a", ["bit.ly/a"]),
    ],
)
def test_urls(text: str, expected: list[str]) -> None:
    assert urls(text) == expected


def test_upi_and_email_are_not_urls() -> None:
    entities = find_entities("Pay kisan@paytm। or write to Help.Desk@Example.co.in", 20)

    assert entities.upi_ids == ["kisan@paytm"]
    assert entities.emails == ["help.desk@example.co.in"]
    assert entities.urls == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call 9876543210", ["+919876543210"]),
        ("call +91 98765 43210 or 09876543210", ["+919876543210"]),
        ("बिजली अधिकारी 0091-9876543210 पर", ["+919876543210"]),
        ("toll free 1800-123-4567", ["18001234567"]),
        ("account 123456789012 and 5876543210", []),
    ],
)
def test_phones(text: str, expected: list[str]) -> None:
    assert find_entities(text, 20).phones == expected


def test_caps_per_kind() -> None:
    text = " ".join(f"98765432{i:02d}" for i in range(10))

    assert len(find_entities(text, 3).phones) == 3


async def test_noop_expander() -> None:
    assert await NoopExpander().expand("bit.ly/x") is None


@pytest.mark.respx(assert_all_called=False)
async def test_redirect_chain_stops_before_the_final_host(respx_mock: respx.MockRouter) -> None:
    respx_mock.head("https://bit.ly/a").respond(301, headers={"Location": "https://tinyurl.com/b"})
    respx_mock.head("https://tinyurl.com/b").respond(302, headers={"Location": "/c"})
    respx_mock.head("https://tinyurl.com/c").respond(
        307, headers={"Location": "http://10.0.0.5/admin"}
    )
    internal = respx_mock.head("http://10.0.0.5/admin").respond(200)

    async with httpx.AsyncClient() as client:
        expanded = await HeadRedirectExpander(client, SHORTENERS, 5).expand("bit.ly/a")

    assert expanded == "http://10.0.0.5/admin"
    assert not internal.called


async def test_non_shortener_is_never_requested(respx_mock: respx.MockRouter) -> None:
    async with httpx.AsyncClient() as client:
        assert await HeadRedirectExpander(client, SHORTENERS, 5).expand("example.com/x") is None

    assert not respx_mock.calls


async def test_expansion_stops_after_max_redirects(respx_mock: respx.MockRouter) -> None:
    respx_mock.head("https://bit.ly/loop").respond(301, headers={"Location": "https://bit.ly/loop"})

    async with httpx.AsyncClient() as client:
        expanded = await HeadRedirectExpander(client, SHORTENERS, 3).expand("https://bit.ly/loop")

    assert expanded == "https://bit.ly/loop"
    assert respx_mock.calls.call_count == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200),
        httpx.Response(301),
        httpx.Response(301, headers={"Location": "ftp://files.example/x"}),
    ],
)
async def test_no_usable_redirect(respx_mock: respx.MockRouter, response: httpx.Response) -> None:
    respx_mock.head("https://bit.ly/x").mock(return_value=response)

    async with httpx.AsyncClient() as client:
        assert await HeadRedirectExpander(client, SHORTENERS, 5).expand("bit.ly/x") is None


async def test_network_error_gives_none(respx_mock: respx.MockRouter) -> None:
    respx_mock.head("https://bit.ly/x").mock(side_effect=httpx.ConnectError("down"))

    async with httpx.AsyncClient() as client:
        assert await HeadRedirectExpander(client, SHORTENERS, 5).expand("bit.ly/x") is None


async def test_invalid_url_gives_none() -> None:
    async with httpx.AsyncClient() as client:
        assert await HeadRedirectExpander(client, SHORTENERS, 5).expand("bit.ly:abc/x") is None


class SlowExpander:
    async def expand(self, url: str) -> str | None:
        await asyncio.sleep(10)
        return "https://never.example"


class MapExpander:
    def __init__(self, targets: dict[str, str]) -> None:
        self.targets = targets
        self.calls: list[str] = []

    async def expand(self, url: str) -> str | None:
        self.calls.append(url)
        return self.targets.get(url)


async def test_extract_expands_only_shorteners() -> None:
    expander = MapExpander({"bit.ly/a": "https://sbi-kyc.example/login"})

    entities = await extract_entities(
        "go to bit.ly/a or example.com", expander=expander, shorteners=SHORTENERS, limits=LIMITS
    )

    assert expander.calls == ["bit.ly/a"]
    assert entities.urls == [
        UrlEntity(url="bit.ly/a", expanded_url="https://sbi-kyc.example/login"),
        UrlEntity(url="example.com"),
    ]


async def test_extract_timeout_keeps_the_original() -> None:
    limits = EntityLimits(max_per_kind=20, expand_timeout_s=0.01)

    entities = await extract_entities(
        "go to bit.ly/a", expander=SlowExpander(), shorteners=SHORTENERS, limits=limits
    )

    assert entities.urls == [UrlEntity(url="bit.ly/a")]
