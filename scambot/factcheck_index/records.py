"""Fact-check records, URL canonicalisation and the passage text that gets embedded."""

import re
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict

from scambot.core.config import SourcesConfig

MAX_CLAIM_CHARS: Final = 1000
MAX_TITLE_CHARS: Final = 500
MAX_RATING_CHARS: Final = 200
MAX_SOURCE_CHARS: Final = 64
MAX_URL_CHARS: Final = 2048

_TRACKING_PARAMS: Final = frozenset({"fbclid", "gclid"})
_LANGUAGE_RE: Final = re.compile(r"^[a-z]{2,3}$")


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArticleRecord(_Record):
    """One fact-check article as stored in the index."""

    source: str
    url: str
    title: str
    claim: str
    rating: str | None = None
    published_at: AwareDatetime | None = None
    language: str | None = None


class ClaimReview(_Record):
    """One review returned by the Fact Check Tools search."""

    claim: str
    claimant: str | None = None
    publisher_name: str | None = None
    publisher_site: str | None = None
    url: str
    title: str | None = None
    rating: str | None = None
    review_date: AwareDatetime | None = None
    language: str | None = None


def _is_tracking(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("utm_") or lowered in _TRACKING_PARAMS


def canonical_url(url: str) -> str:
    """Lower-case scheme and host, no fragment and no tracking parameters."""
    parts = urlsplit(url.strip())
    query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), "")
    )


def normalise_language(value: object) -> str | None:
    """``"en-US"`` → ``"en"``; ``{"alternateName": "hi"}`` → ``"hi"``; anything else → None."""
    if isinstance(value, dict):
        value = value.get("alternateName") or value.get("name")
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-").split("-")[0]
    return code if _LANGUAGE_RE.match(code) else None


def _host(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    return parts.hostname.lower()


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value[:limit] if value else None


def record_from_review(review: ClaimReview, catalogue: SourcesConfig) -> ArticleRecord | None:
    """The index record for a search result; None for a non-web URL or a blank claim."""
    host = _host(review.url)
    claim = _clip(review.claim, MAX_CLAIM_CHARS)
    if host is None or claim is None:
        return None
    known = catalogue.by_host(host)
    source = known.id if known else (_clip(review.publisher_name, MAX_SOURCE_CHARS) or host)
    return ArticleRecord(
        source=source[:MAX_SOURCE_CHARS],
        url=review.url,
        title=_clip(review.title, MAX_TITLE_CHARS) or claim[:MAX_TITLE_CHARS],
        claim=claim,
        rating=_clip(review.rating, MAX_RATING_CHARS),
        published_at=review.review_date,
        language=normalise_language(review.language),
    )


def passage_text(record: ArticleRecord) -> str:
    """The text embedded for an article: its claim, then its title when that differs."""
    if record.title == record.claim:
        return record.claim
    return f"{record.claim}. {record.title}"
