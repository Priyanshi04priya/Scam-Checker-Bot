"""Reading the title, description and publication date of a news page."""

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Any, Final

import trafilatura
from dateutil import parser as dateparser
from lxml import html as lxml_html

from scambot.core.logging import get_logger

TITLE_CHARS: Final = 200
DESCRIPTION_CHARS: Final = 300
_ARTICLE_TYPES: Final = frozenset(
    {
        "newsarticle",
        "article",
        "reportagenewsarticle",
        "webpage",
        "imageobject",
        "videoobject",
        "liveblogposting",
    }
)
_DATE_META_PROPERTIES: Final = (
    "article:published_time",
    "og:article:published_time",
    "og:published_time",
)
_DATE_META_NAMES: Final = (
    "date",
    "pubdate",
    "publishdate",
    "publish-date",
    "dc.date.issued",
    "dc.date",
    "parsely-pub-date",
    "sailthru.date",
    "article.published",
)

_FULL_DATE: Final = re.compile(r"^\d{4}-?\d{2}-?\d{2}")

_log = get_logger(component="page_extract")


@dataclass(frozen=True)
class PageInfo:
    title: str | None
    description: str | None
    publisher: str | None
    published: dt.date | None


def parse_date(value: object) -> dt.date | None:
    """The calendar date as written in an ISO 8601 string (no timezone shift)."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _FULL_DATE.match(text):
        return None  # a bare year or month would count as the first day and skew the age
    candidate = text[:-1] if text.endswith("Z") else text
    try:
        return dt.datetime.fromisoformat(candidate).date()
    except ValueError:
        pass
    try:
        parsed: dt.date = dateparser.isoparse(text).date()
        return parsed
    except (ValueError, OverflowError):
        return None


def _clean(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text[:limit] if text else None


def _meta(tree: Any, key: str, names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for node in tree.iter("meta"):
        label = (node.get(key) or "").strip().lower()
        content = node.get("content")
        if label in names and content:
            values.append(content)
    return values


def _jsonld_objects(tree: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kind = node.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if any(isinstance(k, str) and k.lower() in _ARTICLE_TYPES for k in kinds):
                found.append(node)
            walk(node.get("@graph"))

    for script in tree.iter("script"):
        if (script.get("type") or "").strip().lower() != "application/ld+json":
            continue
        try:
            walk(json.loads(script.text_content()))
        except ValueError:
            continue
    return found


def _first_time(tree: Any) -> list[str]:
    values: list[str] = []
    for node in tree.iter("time"):
        if node.get("datetime") and (
            node.get("pubdate") is not None or node.get("itemprop") == "datePublished"
        ):
            values.append(node.get("datetime"))
    for container in ("article", "main"):
        for holder in tree.iter(container):
            for node in holder.iter("time"):
                if node.get("datetime"):
                    values.append(node.get("datetime"))
    return values


def _trafilatura_meta(html: str, url: str) -> Any:
    try:
        return trafilatura.extract_metadata(html, default_url=url, extensive=False)
    except Exception as exc:  # the library can fail on odd markup; the other sources still work
        _log.info("trafilatura_metadata_failed", error=type(exc).__name__)
        return None


def _publisher(objects: list[dict[str, Any]], tree: Any) -> str | None:
    for obj in objects:
        publisher = obj.get("publisher")
        name = publisher.get("name") if isinstance(publisher, dict) else publisher
        cleaned = _clean(name, TITLE_CHARS)
        if cleaned:
            return cleaned
    names = _meta(tree, "property", ("og:site_name",))
    return _clean(names[0], TITLE_CHARS) if names else None


def extract_page(html: str, url: str, *, min_year: int, today: dt.date) -> PageInfo:
    """Best-effort page details. Dates outside ``min_year``..tomorrow are discarded."""
    try:
        tree = lxml_html.fromstring(html)
    except (ValueError, lxml_html.etree.ParserError):
        return PageInfo(None, None, None, None)
    objects = _jsonld_objects(tree)
    ordered: list[object] = [o.get("datePublished") for o in objects]
    ordered += [o.get("dateCreated") for o in objects]
    ordered += _meta(tree, "property", _DATE_META_PROPERTIES)
    ordered += _meta(tree, "name", _DATE_META_NAMES)
    ordered += _meta(tree, "itemprop", ("datepublished",))
    ordered += _first_time(tree)
    meta = None
    published = _first_valid(ordered, min_year, today)
    if published is None:
        meta = _trafilatura_meta(html, url)
        published = _first_valid([getattr(meta, "date", None)], min_year, today)
    titles: list[object] = [o.get("headline") for o in objects]
    titles += _meta(tree, "property", ("og:title",))
    title_node = tree.find(".//title")
    titles.append(title_node.text_content() if title_node is not None else None)
    descriptions: list[object] = [*_meta(tree, "property", ("og:description",))]
    descriptions += _meta(tree, "name", ("description",))
    title = next((t for t in (_clean(v, TITLE_CHARS) for v in titles) if t), None)
    description = next((d for d in (_clean(v, DESCRIPTION_CHARS) for v in descriptions) if d), None)
    if title is None or description is None:
        meta = meta or _trafilatura_meta(html, url)
        title = title or _clean(getattr(meta, "title", None), TITLE_CHARS)
        description = description or _clean(getattr(meta, "description", None), DESCRIPTION_CHARS)
    return PageInfo(title, description, _publisher(objects, tree), published)


def _first_valid(values: list[object], min_year: int, today: dt.date) -> dt.date | None:
    latest = today + dt.timedelta(days=1)
    for value in values:
        parsed = parse_date(value)
        if parsed is not None and parsed.year >= min_year and parsed <= latest:
            return parsed
    return None
