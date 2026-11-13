"""Google Cloud Vision web detection: pages that contain copies of an image."""

import base64
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

import httpx
from pydantic import SecretStr

ANNOTATE_URL: Final = "https://vision.googleapis.com/v1/images:annotate"


class VisionError(Exception):
    """The search failed; the message is a short code such as ``status_403``."""


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str | None
    match: Literal["full", "partial"]


class WebSearch(Protocol):
    async def pages_with_matches(self, image: bytes, *, max_results: int) -> list[WebPage]:
        """Pages that show this image (fully or in part), without duplicates."""
        ...


def _pages(data: Any) -> list[WebPage]:
    response = data["responses"][0]
    error = response.get("error")
    if error:
        code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
        msg = f"vision_error_{code}"
        raise VisionError(msg)
    found: dict[str, WebPage] = {}
    for entry in response.get("webDetection", {}).get("pagesWithMatchingImages", []):
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            continue
        title = entry.get("pageTitle")
        title = title if isinstance(title, str) else None
        match: Literal["full", "partial"] = "full" if entry.get("fullMatchingImages") else "partial"
        known = found.get(url)
        if known is None or (known.match == "partial" and match == "full"):
            found[url] = WebPage(url, title or (known.title if known else None), match)
    return list(found.values())


class VisionClient:
    """The key travels in a header, never in the URL."""

    def __init__(self, http: httpx.AsyncClient, api_key: SecretStr) -> None:
        self._http = http
        self._api_key = api_key

    async def pages_with_matches(self, image: bytes, *, max_results: int) -> list[WebPage]:
        body = {
            "requests": [
                {
                    "image": {"content": base64.b64encode(image).decode("ascii")},
                    "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
                }
            ]
        }
        try:
            response = await self._http.post(
                ANNOTATE_URL,
                json=body,
                headers={"X-Goog-Api-Key": self._api_key.get_secret_value()},
            )
        except httpx.HTTPError as exc:
            msg = f"request_failed_{type(exc).__name__}"
            raise VisionError(msg) from exc
        if not response.is_success:
            msg = f"status_{response.status_code}"
            raise VisionError(msg)
        try:
            return _pages(response.json())
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            msg = "unexpected_response"
            raise VisionError(msg) from exc
