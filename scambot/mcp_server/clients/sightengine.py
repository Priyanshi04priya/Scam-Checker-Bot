"""Sightengine image analysis: AI-generated and deepfake scores.

The response fields are read tolerantly: a missing or malformed score becomes ``None``.
"""

from dataclasses import dataclass
from typing import Any, Final, Protocol

import httpx
from pydantic import SecretStr

CHECK_URL: Final = "https://api.sightengine.com/1.0/check.json"
MODELS: Final = "genai,deepfake"


class SightengineError(Exception):
    """The analysis failed; the message is a short code such as ``status_401``."""


@dataclass(frozen=True)
class SynthScores:
    ai_generated: float | None
    deepfake: float | None


class ImageAnalyzer(Protocol):
    async def scores(self, image: bytes, filename: str) -> SynthScores: ...


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return min(1.0, max(0.0, float(value)))


class SightengineClient:
    """Credentials go in the form body, never in the URL."""

    def __init__(self, http: httpx.AsyncClient, user: str, secret: SecretStr) -> None:
        self._http = http
        self._user = user
        self._secret = secret

    async def scores(self, image: bytes, filename: str) -> SynthScores:
        try:
            response = await self._http.post(
                CHECK_URL,
                files={"media": (filename, image)},
                data={
                    "models": MODELS,
                    "api_user": self._user,
                    "api_secret": self._secret.get_secret_value(),
                },
            )
        except httpx.HTTPError as exc:
            msg = f"request_failed_{type(exc).__name__}"
            raise SightengineError(msg) from exc
        if not response.is_success:
            msg = f"status_{response.status_code}"
            raise SightengineError(msg)
        try:
            data: Any = response.json()
            if data.get("status") != "success":
                error = data.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                msg = f"vendor_error_{code}" if code is not None else "vendor_failure"
                raise SightengineError(msg)
            kind = data.get("type")
            kind = kind if isinstance(kind, dict) else {}
            return SynthScores(_score(kind.get("ai_generated")), _score(kind.get("deepfake")))
        except (ValueError, AttributeError) as exc:
            msg = "unexpected_response"
            raise SightengineError(msg) from exc
