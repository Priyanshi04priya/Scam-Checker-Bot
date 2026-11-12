"""LLM client interface, errors and the client used when no provider key is configured."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel

from scambot.core.settings import Effort

__all__ = [
    "DisabledLLMClient",
    "Effort",
    "ImageMediaType",
    "LLMCallConfig",
    "LLMClient",
    "LLMError",
    "LLMImage",
    "LLMRefusalError",
    "LLMUnavailableError",
]

ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class LLMError(Exception):
    """The call failed or its reply could not be used."""


class LLMUnavailableError(LLMError):
    """No LLM is configured."""


class LLMRefusalError(LLMError):
    """The model declined to answer."""


@dataclass(frozen=True)
class LLMImage:
    media_type: ImageMediaType
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class LLMCallConfig:
    model: str
    effort: Effort
    max_tokens: int


class LLMClient(Protocol):
    """Structured LLM calls: a named prompt plus variables in, a validated model out."""

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T: ...


class DisabledLLMClient:
    """Used when no provider key is set: every call fails with ``LLMUnavailableError``."""

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T:
        msg = "llm not configured"
        raise LLMUnavailableError(msg)
