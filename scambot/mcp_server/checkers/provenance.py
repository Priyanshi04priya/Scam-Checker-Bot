"""Finding where a picture first appeared.

The picture is searched on the web, only pages of listed publishers are read, and the oldest
dated page decides whether the picture is old. Embedded camera data and content credentials are
reported but never decide the result, because they can be removed or forged.
"""

import asyncio
import datetime as dt
import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal
from urllib.parse import urlsplit

from PIL import Image, ImageOps

from scambot.core.config import ProvenanceThresholds, SourcesConfig
from scambot.core.contracts import CheckerResult, Evidence, Signal
from scambot.core.logging import get_logger
from scambot.core.media_store import ORIGINAL
from scambot.mcp_server.clients.pagefetch import FetchError, PageFetch
from scambot.mcp_server.clients.vision import VisionError, WebPage, WebSearch
from scambot.mcp_server.media import MediaFiles, MediaMissing
from scambot.mcp_server.page_extract import PageInfo, extract_page

CHECKER: Final = "check_provenance"
MAX_IMAGE_BYTES: Final = 30_000_000
_EXIF_DATE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})")
_MAX_ACTIONS: Final = 10

_log = get_logger(component="provenance")


def utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


@dataclass(frozen=True)
class ExifInfo:
    captured_at: dt.date | None = None
    device: str | None = None
    software: str | None = None

    def as_details(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "device": self.device,
            "software": self.software,
        }


def _exif_date(value: object) -> dt.date | None:
    match = _EXIF_DATE.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    try:
        return dt.date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None


def _exif_text(value: object) -> str | None:
    return value.strip()[:100] or None if isinstance(value, str) else None


def read_exif(data: bytes) -> ExifInfo:
    """Camera data from the picture; anything unreadable is left out."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            exif = image.getexif()
            sub = exif.get_ifd(0x8769)
            captured = _exif_date(sub.get(36867)) or _exif_date(exif.get(0x0132))
            make, model = _exif_text(exif.get(0x010F)), _exif_text(exif.get(0x0110))
            device = " ".join(part for part in (make, model) if part) or None
            return ExifInfo(captured, device, _exif_text(exif.get(0x0131)))
    except Exception:  # damaged metadata is common and never an error for the check
        return ExifInfo()


def read_c2pa(data: bytes, mime: str) -> dict[str, Any]:
    """Content credentials, when the optional library is installed."""
    try:
        import c2pa  # noqa: PLC0415 - optional dependency
    except ImportError:
        return {"status": "unavailable"}
    try:
        reader = c2pa.Reader(mime, io.BytesIO(data))
        manifest_json = reader.json()
    except Exception as exc:
        if str(exc).startswith("ManifestNotFound"):
            return {"status": "absent"}
        return {"status": "failed"}
    try:
        import json  # noqa: PLC0415

        store = json.loads(manifest_json)
        active = store.get("manifests", {}).get(store.get("active_manifest"), {})
        actions: list[str] = []
        for assertion in active.get("assertions", []):
            if str(assertion.get("label", "")).startswith("c2pa.actions"):
                for action in assertion.get("data", {}).get("actions", []):
                    if isinstance(action, dict) and isinstance(action.get("action"), str):
                        actions.append(action["action"])
        issuer = (active.get("signature_info") or {}).get("issuer")
        state = store.get("validation_state")
        return {
            "status": "present",
            "valid": state in (None, "Valid", "Trusted"),
            "issuer": issuer if isinstance(issuer, str) else None,
            "actions": actions[:_MAX_ACTIONS],
        }
    except (ValueError, AttributeError, TypeError):
        return {"status": "failed"}


def _search_image(data: bytes, max_side: int, quality: int) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail((max_side, max_side))
    out = io.BytesIO()
    prepared.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _mime_of(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return Image.MIME.get(image.format or "", "image/jpeg")
    except Exception:
        return "image/jpeg"


@dataclass
class _Candidate:
    page: WebPage
    host: str
    publisher: str
    order: int
    info: PageInfo | None = None
    error: str | None = None


@dataclass
class _Embedded:
    exif: ExifInfo = field(default_factory=ExifInfo)
    c2pa: dict[str, Any] = field(default_factory=dict)


class ProvenanceChecker:
    def __init__(  # noqa: PLR0913
        self,
        search: WebSearch | None,
        fetcher: PageFetch,
        media: MediaFiles,
        thresholds: ProvenanceThresholds,
        sources: SourcesConfig,
        *,
        today: Callable[[], dt.date] = utc_today,
    ) -> None:
        self._search = search
        self._fetcher = fetcher
        self._media = media
        self._limits = thresholds
        self._sources = sources
        self._today = today

    # --- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _result(  # noqa: PLR0913
        status: Literal["ok", "inconclusive", "error"],
        details: dict[str, Any],
        *,
        signal: Signal = "none",
        confidence: float = 0.0,
        evidence: list[Evidence] | None = None,
        error: str | None = None,
    ) -> CheckerResult:
        return CheckerResult(
            checker=CHECKER,
            status=status,
            signal=signal,
            confidence=confidence,
            evidence=evidence or [],
            duration_ms=0,
            error=error,
            details=details,
        )

    def _candidates(self, pages: list[WebPage]) -> list[_Candidate]:
        found: list[_Candidate] = []
        for order, page in enumerate(pages):
            host = urlsplit(page.url).hostname or ""
            publisher = self._sources.credible_name(host) if host else None
            if publisher is not None:
                found.append(_Candidate(page, host, publisher, order))
        found.sort(key=lambda c: (c.page.match != "full", c.order))
        return found[: self._limits.max_pages]

    async def _fetch(self, candidate: _Candidate, today: dt.date) -> None:
        try:
            html = await self._fetcher.get_html(candidate.page.url)
            candidate.info = await asyncio.to_thread(
                extract_page, html, candidate.page.url, min_year=self._limits.min_year, today=today
            )
        except FetchError as exc:
            candidate.error = str(exc)
        except Exception as exc:  # a page that cannot be parsed is dropped
            candidate.error = type(exc).__name__

    async def _fetch_all(self, candidates: list[_Candidate], today: dt.date) -> None:
        try:
            async with asyncio.timeout(self._limits.pages_timeout_s):
                await asyncio.gather(*(self._fetch(c, today) for c in candidates))
        except TimeoutError:
            for candidate in candidates:
                if candidate.info is None and candidate.error is None:
                    candidate.error = "timeout"

    @staticmethod
    def _page_details(candidates: list[_Candidate]) -> list[dict[str, Any]]:
        return [
            {
                "host": c.host,
                "date": c.info.published.isoformat() if c.info and c.info.published else None,
                "error": c.error,
            }
            for c in candidates
        ]

    def _evidence(self, dated: list[_Candidate], origin: _Candidate) -> list[Evidence]:
        others = [c for c in dated if c is not origin]
        others.sort(key=lambda c: (_date(c), c.order))
        evidence: list[Evidence] = []
        for candidate in [origin, *others][: self._limits.max_evidence]:
            info = candidate.info
            assert info is not None  # noqa: S101 - dated candidates always have details
            title = info.title or candidate.page.title or "Photo"
            evidence.append(
                Evidence(
                    summary=f"{candidate.publisher}: {title} ({info.published})",
                    url=candidate.page.url,
                    source=candidate.publisher,
                    date=info.published,
                )
            )
        return evidence

    async def _embedded(self, data: bytes) -> _Embedded:
        exif = await asyncio.to_thread(read_exif, data)
        c2pa = await asyncio.to_thread(read_c2pa, data, _mime_of(data))
        return _Embedded(exif, c2pa)

    async def _find_pages(
        self, data: bytes, details: dict[str, Any]
    ) -> list[WebPage] | CheckerResult:
        if self._search is None:
            return self._result("error", details, error="not_configured")
        try:
            prepared = await asyncio.to_thread(
                _search_image, data, self._limits.max_image_side_px, self._limits.jpeg_quality
            )
        except Exception:
            return self._result("error", details, error="unreadable_image")
        try:
            pages = await asyncio.wait_for(
                self._search.pages_with_matches(prepared, max_results=self._limits.max_results),
                self._limits.vision_timeout_s,
            )
        except VisionError as exc:
            return self._result("error", details, error=str(exc))
        except TimeoutError:
            return self._result("error", details, error="timeout")
        return pages

    # --- check -----------------------------------------------------------------------------

    async def check(
        self, media_key: str, reference_date: dt.date | None, image_name: str = ORIGINAL
    ) -> CheckerResult:
        """Look for the oldest credible page showing this picture (a video's key frame)."""
        try:
            data = self._media.read(media_key, image_name, max_bytes=MAX_IMAGE_BYTES)
        except MediaMissing as exc:
            return self._result("error", {}, error=str(exc))
        embedded = await self._embedded(data)
        details: dict[str, Any] = {
            "exif": embedded.exif.as_details(),
            "c2pa": embedded.c2pa,
        }
        found = await self._find_pages(data, details)
        if isinstance(found, CheckerResult):
            return found
        if not found:
            return self._result("inconclusive", {**details, "reason": "no_match"})
        return await self._conclude(found, details, reference_date)

    async def _conclude(
        self, pages: list[WebPage], details: dict[str, Any], reference_date: dt.date | None
    ) -> CheckerResult:
        candidates = self._candidates(pages)
        details["candidates"] = len(candidates)
        if not candidates:
            return self._result("inconclusive", {**details, "reason": "no_credible_page"})
        today = self._today()
        await self._fetch_all(candidates, today)
        details["pages"] = self._page_details(candidates)
        dated = [c for c in candidates if c.info is not None and c.info.published is not None]
        if not dated:
            return self._result("inconclusive", {**details, "reason": "no_dated_page"})
        origin = min(dated, key=lambda c: (_date(c), c.page.match != "full", c.order))
        assert origin.info is not None  # noqa: S101
        published = _date(origin)
        reference = reference_date or today
        age_days = (reference - published).days
        full = origin.page.match == "full"
        limits = self._limits
        old = age_days > limits.old_media_days
        if old:
            signal: Signal = "old_media"
            confidence = limits.confidence_old_full if full else limits.confidence_old_partial
        else:
            signal = "none"
            confidence = limits.confidence_clear_full if full else limits.confidence_clear_partial
        details["reference_date"] = reference.isoformat()
        details["origin"] = {
            "url": origin.page.url,
            "publisher": origin.publisher,
            "domain": origin.host,
            "title": origin.info.title or origin.page.title,
            "description": origin.info.description,
            "date": published.isoformat(),
            "match": origin.page.match,
            "credible": True,
            "age_days": age_days,
        }
        return self._result(
            "ok",
            details,
            signal=signal,
            confidence=confidence,
            evidence=self._evidence(dated, origin),
        )


def _date(candidate: _Candidate) -> dt.date:
    assert candidate.info is not None  # noqa: S101
    assert candidate.info.published is not None  # noqa: S101
    return candidate.info.published
