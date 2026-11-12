"""Google Fact Check Tools (``claims:search``) client."""

from typing import Any, Final

import httpx
import pydantic
from pydantic import SecretStr

from scambot.factcheck_index.records import ClaimReview

SEARCH_URL: Final = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


class FactCheckToolsError(Exception):
    """The search failed; the message is a short code such as ``status_503``."""


def _reviews(data: Any) -> list[ClaimReview]:
    reviews: list[ClaimReview] = []
    for claim in data.get("claims", []):
        text = claim.get("text")
        for review in claim.get("claimReview", []):
            url = review.get("url")
            if not text or not url:
                continue
            publisher = review.get("publisher") or {}
            reviews.append(
                ClaimReview(
                    claim=text,
                    claimant=claim.get("claimant"),
                    publisher_name=publisher.get("name"),
                    publisher_site=publisher.get("site"),
                    url=url,
                    title=review.get("title"),
                    rating=review.get("textualRating"),
                    review_date=review.get("reviewDate"),
                    language=review.get("languageCode"),
                )
            )
    return reviews


class FactCheckToolsClient:
    """The key travels in a header, never in the URL, so request logs cannot leak it."""

    def __init__(self, http: httpx.AsyncClient, api_key: SecretStr) -> None:
        self._http = http
        self._api_key = api_key

    async def _page(self, params: dict[str, str | int]) -> tuple[list[ClaimReview], str | None]:
        try:
            response = await self._http.get(
                SEARCH_URL,
                params=params,
                headers={"X-Goog-Api-Key": self._api_key.get_secret_value()},
            )
        except httpx.HTTPError as exc:
            msg = f"request_failed_{type(exc).__name__}"
            raise FactCheckToolsError(msg) from exc
        if not response.is_success:
            msg = f"status_{response.status_code}"
            raise FactCheckToolsError(msg)
        try:
            data: Any = response.json()
            token = data.get("nextPageToken")
            return _reviews(data), token if isinstance(token, str) and token else None
        except (ValueError, KeyError, TypeError, AttributeError, pydantic.ValidationError) as exc:
            msg = "unexpected_response"
            raise FactCheckToolsError(msg) from exc

    async def search(self, query: str, *, page_size: int) -> list[ClaimReview]:
        """Reviews of claims matching ``query``."""
        reviews, _token = await self._page({"query": query, "pageSize": page_size})
        return reviews

    async def by_publisher(
        self, site: str, *, max_age_days: int, page_size: int, max_pages: int
    ) -> list[ClaimReview]:
        """Recent reviews published by ``site``, following page tokens up to ``max_pages``."""
        params: dict[str, str | int] = {
            "reviewPublisherSiteFilter": site,
            "maxAgeDays": max_age_days,
            "pageSize": page_size,
        }
        reviews: list[ClaimReview] = []
        for _ in range(max_pages):
            page, token = await self._page(params)
            reviews.extend(page)
            if token is None:
                break
            params["pageToken"] = token
        return reviews
