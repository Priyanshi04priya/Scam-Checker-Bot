"""Links, phone numbers, UPI IDs and e-mail addresses in message text."""

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import urljoin

import httpx

from scambot.core.contracts import Entities, UrlEntity
from scambot.core.domains import (
    absolute_url,
    domain_label,
    has_public_suffix,
    registrable_domain,
    url_host,
)
from scambot.core.logging import get_logger

_log = get_logger(component="entities")

_EMAIL_RE: Final = re.compile(
    r"(?<![\w.+-])[a-z0-9][a-z0-9._%+-]{0,63}@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,24}(?![\w-])",
    re.IGNORECASE,
)
# A UPI handle (the part after @) has no dot, which tells it apart from an e-mail address.
_UPI_RE: Final = re.compile(
    r"(?<![\w.+-])[a-z0-9][a-z0-9._-]{1,63}@[a-z][a-z0-9]{1,31}(?![\w.@-])", re.IGNORECASE
)
_SCHEMED_RE: Final = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'`]+", re.IGNORECASE)
_BARE_RE: Final = re.compile(
    r"(?<![@\w.-])(?:[a-z0-9¡-￿](?:[a-z0-9¡-￿-]{0,61}[a-z0-9¡-￿])?\.)+"
    r"(?:[a-z¡-￿]{2,24}|xn--[a-z0-9-]{1,59})(?:/[^\s<>\"'`]*)?(?![\w@-])",
    re.IGNORECASE,
)
_PHONE_RE: Final = re.compile(
    r"(?<![\d+])(?:(?:\+|00)91[\s-]?|0)?([6-9]\d{4})[\s-]?(\d{5})(?!\d)"
    r"|(?<!\d)(1800)[\s-]?(\d{3})[\s-]?(\d{4})(?!\d)"
)
_TRAILING: Final = ".,;:!?)]}>'\"।"
_REDIRECT_STATUSES: Final = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True)
class EntityLimits:
    max_per_kind: int
    expand_timeout_s: float


def _dedupe(items: Iterable[str], limit: int) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
            if len(seen) == limit:
                break
    return list(seen)


def _blank(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _is_bare_domain(candidate: str) -> bool:
    host = url_host(candidate)
    return bool(host) and has_public_suffix(host or "") and not domain_label(host or "").isdigit()


def _find_urls(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    schemed: list[tuple[int, int]] = []
    for match in _SCHEMED_RE.finditer(text):
        schemed.append(match.span())
        found.append((match.start(), match.group(0).rstrip(_TRAILING)))
    for match in _BARE_RE.finditer(text):
        if any(start <= match.start() < end for start, end in schemed):
            continue
        candidate = match.group(0).rstrip(_TRAILING)
        if _is_bare_domain(candidate):
            found.append((match.start(), candidate))
    return [url for _, url in sorted(found)]


def _phones(text: str) -> list[str]:
    phones: list[str] = []
    for match in _PHONE_RE.finditer(text):
        if match.group(1):
            phones.append(f"+91{match.group(1)}{match.group(2)}")
        else:
            phones.append(f"{match.group(3)}{match.group(4)}{match.group(5)}")
    return phones


def find_entities(text: str, max_per_kind: int) -> Entities:
    """Every entity in ``text``, in order of appearance, deduplicated and capped per kind."""
    emails = [m.group(0) for m in _EMAIL_RE.finditer(text)]
    upi_matches = list(_UPI_RE.finditer(text))
    upi_ids = [m.group(0).rstrip(_TRAILING) for m in upi_matches]
    taken = [m.span() for m in _EMAIL_RE.finditer(text)] + [m.span() for m in upi_matches]
    urls = _find_urls(_blank(text, taken))
    return Entities(
        urls=[UrlEntity(url=url) for url in _dedupe(urls, max_per_kind)],
        phones=_dedupe(_phones(text), max_per_kind),
        upi_ids=_dedupe((u.lower() for u in upi_ids), max_per_kind),
        emails=_dedupe((e.lower() for e in emails), max_per_kind),
    )


class UrlExpander(Protocol):
    async def expand(self, url: str) -> str | None:
        """Where a shortened link leads, or None when that cannot be found out."""
        ...


class NoopExpander:
    """Never expands anything (expansion switched off)."""

    async def expand(self, url: str) -> str | None:
        return None


class HeadRedirectExpander:
    """Follows redirects by hand, and only while they stay on known shortener hosts.

    The final host is never requested, so a link cannot make the service call an internal or
    attacker-chosen address.
    """

    def __init__(
        self, client: httpx.AsyncClient, shorteners: frozenset[str], max_redirects: int
    ) -> None:
        self._client = client
        self._shorteners = shorteners
        self._max_redirects = max_redirects

    def _is_shortener(self, url: str) -> bool:
        host = url_host(url)
        return host is not None and registrable_domain(host) in self._shorteners

    async def expand(self, url: str) -> str | None:
        current = absolute_url(url)
        if not self._is_shortener(current):
            return None
        target: str | None = None
        for _hop in range(self._max_redirects):
            try:
                response = await self._client.head(current, follow_redirects=False)
            except (httpx.HTTPError, httpx.InvalidURL) as exc:  # InvalidURL is no HTTPError
                _log.info("url_expand_failed", error=type(exc).__name__)
                return None
            location = response.headers.get("location")
            if response.status_code not in _REDIRECT_STATUSES or not location:
                break
            following = urljoin(current, location)
            if url_host(following) is None or "://" not in following:
                break
            target = following
            if not self._is_shortener(following):
                break
            current = following
        return target


async def extract_entities(
    text: str, *, expander: UrlExpander, shorteners: frozenset[str], limits: EntityLimits
) -> Entities:
    """Find entities, then expand shortened links concurrently within a time limit."""
    entities = find_entities(text, limits.max_per_kind)
    short = [
        index
        for index, entity in enumerate(entities.urls)
        if (host := url_host(entity.url)) is not None and registrable_domain(host) in shorteners
    ]
    if not short:
        return entities
    expanded: list[str | None] = [None] * len(short)

    async def expand_one(slot: int, url: str) -> None:
        expanded[slot] = await expander.expand(url)

    try:
        async with asyncio.timeout(limits.expand_timeout_s), asyncio.TaskGroup() as group:
            for slot, index in enumerate(short):
                group.create_task(expand_one(slot, entities.urls[index].url))
    except TimeoutError:
        _log.info("url_expand_timeout", urls=len(short))
    urls = list(entities.urls)
    for slot, index in enumerate(short):
        if expanded[slot]:
            urls[index] = urls[index].model_copy(update={"expanded_url": expanded[slot]})
    return entities.model_copy(update={"urls": urls})
