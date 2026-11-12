"""Google Safe Browsing (Lookup API v4) client."""

from collections.abc import Sequence
from typing import Any, Final, Protocol

import httpx
from pydantic import SecretStr

FIND_URL: Final = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
THREAT_TYPES: Final = (
    "MALWARE",
    "SOCIAL_ENGINEERING",
    "UNWANTED_SOFTWARE",
    "POTENTIALLY_HARMFUL_APPLICATION",
)


class SafeBrowsingError(Exception):
    """The lookup failed; the message is a short fixed description."""


class UrlReputation(Protocol):
    async def unsafe_urls(self, urls: Sequence[str]) -> dict[str, str]:
        """Map each listed unsafe URL to its threat type; safe URLs are left out."""
        ...


class SafeBrowsingClient:
    """The key travels in a header, never in the URL, so request logs cannot leak it."""

    def __init__(self, http: httpx.AsyncClient, api_key: SecretStr) -> None:
        self._http = http
        self._api_key = api_key

    async def unsafe_urls(self, urls: Sequence[str]) -> dict[str, str]:
        if not urls:
            return {}
        body = {
            "client": {"clientId": "scambot", "clientVersion": "0.1.0"},
            "threatInfo": {
                "threatTypes": list(THREAT_TYPES),
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url} for url in urls],
            },
        }
        try:
            response = await self._http.post(
                FIND_URL, json=body, headers={"X-Goog-Api-Key": self._api_key.get_secret_value()}
            )
        except httpx.HTTPError as exc:
            msg = f"request failed: {type(exc).__name__}"
            raise SafeBrowsingError(msg) from exc
        if not response.is_success:
            msg = f"status {response.status_code}"
            raise SafeBrowsingError(msg)
        try:
            data: Any = response.json()
            return {
                match["threat"]["url"]: match["threatType"] for match in data.get("matches", [])
            }
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            msg = "unexpected response"
            raise SafeBrowsingError(msg) from exc
