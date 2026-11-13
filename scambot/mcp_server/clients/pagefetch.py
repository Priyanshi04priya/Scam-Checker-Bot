"""Fetching a public web page safely.

Every address a host resolves to must be a global one, the connection goes to the address that
was checked (so a second DNS answer cannot redirect it), and redirects are followed by hand so
each hop is checked again.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from typing import Final, Protocol
from urllib.parse import urljoin, urlsplit

import httpx

from scambot.core.config import FetchThresholds
from scambot.core.logging import get_logger

USER_AGENT: Final = "scambot/0.1 (+media provenance check)"
_ALLOWED_PORTS: Final = frozenset({None, 80, 443})
_HTML_TYPES: Final = ("text/html", "application/xhtml+xml")
_BLOCKED_SUFFIXES: Final = (".local", ".internal", ".localhost")
_REDIRECTS: Final = frozenset({301, 302, 303, 307, 308})

_log = get_logger(component="pagefetch")

type Resolver = Callable[[str], Awaitable[list[str]]]


class FetchError(Exception):
    """The page was not fetched; the message is a short code such as ``host_not_public``."""


class PageFetch(Protocol):
    async def get_html(self, url: str) -> str: ...


async def system_resolver(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_v4(ip)
        if embedded is not None:
            return embedded.is_global
    return ip.is_global


_NAT64: Final = ipaddress.IPv6Network("64:ff9b::/96")


def _embedded_v4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 one wraps (mapped, 6to4, Teredo, NAT64), which decides its reach."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def _literal_ip(host: str) -> str | None:
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def _fail(code: str) -> FetchError:
    return FetchError(code)


class SafePageFetcher:
    def __init__(
        self,
        http: httpx.AsyncClient,
        limits: FetchThresholds,
        *,
        resolver: Resolver = system_resolver,
    ) -> None:
        self._http = http
        self._limits = limits
        self._resolver = resolver

    async def _pinned_address(self, url: str) -> tuple[str, str, str]:
        """Validate ``url``; return the host, the address to connect to and the pinned URL."""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise _fail("scheme")
        try:
            port = parts.port
        except ValueError as exc:
            raise _fail("port") from exc
        if port not in _ALLOWED_PORTS:
            raise _fail("port")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise _fail("userinfo")
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            raise _fail("host_not_public")
        literal = _literal_ip(host)
        if literal is None and (
            "." not in host or host == "localhost" or host.endswith(_BLOCKED_SUFFIXES)
        ):
            raise _fail("host_not_public")
        if literal is not None:
            addresses = [literal]
        else:
            try:
                addresses = await self._resolver(host)
            except OSError as exc:
                raise _fail("dns_failed") from exc
            if not addresses:
                raise _fail("dns_failed")
        if not all(_is_public(address) for address in addresses):
            raise _fail("host_not_public")
        address = addresses[0]
        shown = f"[{address}]" if ":" in address else address
        netloc = shown if port is None else f"{shown}:{port}"
        pinned = parts._replace(netloc=netloc).geturl()
        return host, address, pinned

    async def _read(self, host: str, pinned: str) -> tuple[str | None, str | None]:
        """One request: ``(html, None)`` for a page, ``(None, location)`` for a redirect."""
        headers = {"Host": host, "User-Agent": USER_AGENT, "Accept": "text/html"}
        async with self._http.stream(
            "GET",
            pinned,
            headers=headers,
            extensions={"sni_hostname": host},
            follow_redirects=False,
        ) as response:
            if response.status_code in _REDIRECTS:
                return None, response.headers.get("location") or ""
            if response.status_code != 200:
                raise _fail(f"status_{response.status_code}")
            kind = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if kind not in _HTML_TYPES:
                raise _fail("not_html")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > self._limits.max_bytes:
                    raise _fail("too_large")
            try:
                return bytes(body).decode(response.encoding or "utf-8", errors="replace"), None
            except LookupError:
                return bytes(body).decode("utf-8", errors="replace"), None

    async def _get(self, url: str) -> str:
        current = url
        for _hop in range(self._limits.max_redirects + 1):
            host, _address, pinned = await self._pinned_address(current)
            html, location = await self._read(host, pinned)
            if html is not None:
                return html
            if not location:
                raise _fail("request_failed")
            current = urljoin(current, location)
        raise _fail("redirect_limit")

    async def get_html(self, url: str) -> str:
        try:
            async with asyncio.timeout(self._limits.timeout_s):
                return await self._get(url)
        except FetchError as exc:
            _log.info("page_fetch_failed", host=urlsplit(url).hostname, error=str(exc))
            raise
        except TimeoutError as exc:
            raise _fail("timeout") from exc
        except httpx.TimeoutException as exc:
            raise _fail("timeout") from exc
        except httpx.HTTPError as exc:
            raise _fail("request_failed") from exc
