"""Structured calls to the Anthropic Messages API."""

import base64
import time
from collections.abc import Mapping, Sequence
from typing import Final

import anthropic
import pydantic
from anthropic import AsyncAnthropic
from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaImageBlockParam,
    BetaTextBlockParam,
    BetaUserMessageParam,
)
from pydantic import BaseModel

from scambot.core.llm.base import (
    DisabledLLMClient,
    LLMCallConfig,
    LLMClient,
    LLMError,
    LLMImage,
    LLMRefusalError,
)
from scambot.core.llm.prompts import PromptLibrary
from scambot.core.llm.usage import LLMUsage, record_usage
from scambot.core.logging import get_logger
from scambot.core.settings import Settings

CALL_NAMES: Final = ("analyze", "scam_classify", "explanation", "factcheck_relevance")
FALLBACK_BETA: Final = "server-side-fallback-2026-07-01"

_log = get_logger(component="llm")


def call_configs(settings: Settings) -> dict[str, LLMCallConfig]:
    """Model, effort and token budget for each named call."""
    return {
        "analyze": LLMCallConfig(
            settings.llm_model_analyze,
            settings.llm_effort_analyze,
            settings.llm_max_tokens_analyze,
        ),
        "scam_classify": LLMCallConfig(
            settings.llm_model_scam_classify,
            settings.llm_effort_scam_classify,
            settings.llm_max_tokens_scam_classify,
        ),
        "explanation": LLMCallConfig(
            settings.llm_model_explanation,
            settings.llm_effort_explanation,
            settings.llm_max_tokens_explanation,
        ),
        "factcheck_relevance": LLMCallConfig(
            settings.llm_model_factcheck_relevance,
            settings.llm_effort_factcheck_relevance,
            settings.llm_max_tokens_factcheck_relevance,
        ),
    }


def _message(user: str, images: Sequence[LLMImage] | None) -> BetaUserMessageParam:
    """The user turn: images first, then the rendered prompt text."""
    blocks: list[BetaImageBlockParam | BetaTextBlockParam] = [
        BetaImageBlockParam(
            type="image",
            source=BetaBase64ImageSourceParam(
                type="base64",
                media_type=image.media_type,
                data=base64.b64encode(image.data).decode("ascii"),
            ),
        )
        for image in images or ()
    ]
    blocks.append(BetaTextBlockParam(type="text", text=user))
    return BetaUserMessageParam(role="user", content=blocks)


class AnthropicLLMClient:
    """``LLMClient`` backed by the Messages API with JSON-schema structured output."""

    def __init__(
        self,
        client: AsyncAnthropic,
        prompts: PromptLibrary,
        calls: Mapping[str, LLMCallConfig],
        *,
        refusal_fallback: bool,
    ) -> None:
        missing = sorted(set(calls) - prompts.names)
        if missing:
            msg = f"no prompt file for LLM calls: {', '.join(missing)}"
            raise ValueError(msg)
        self._client = client
        self._prompts = prompts
        self._calls = dict(calls)
        self._refusal_fallback = refusal_fallback

    async def structured[T: BaseModel](
        self,
        prompt_name: str,
        variables: Mapping[str, str],
        schema: type[T],
        images: Sequence[LLMImage] | None = None,
    ) -> T:
        config = self._calls.get(prompt_name)
        if config is None:
            msg = f"no call configuration for {prompt_name!r}"
            raise LLMError(msg)
        system, user = self._prompts.render(prompt_name, variables)
        fallback = self._refusal_fallback
        started = time.perf_counter()
        try:
            message = await self._client.beta.messages.parse(
                model=config.model,
                max_tokens=config.max_tokens,
                system=system,
                messages=[_message(user, images)],
                output_format=schema,
                output_config={"effort": config.effort},
                betas=[FALLBACK_BETA] if fallback else anthropic.omit,
                fallbacks="default" if fallback else anthropic.omit,
            )
        except (anthropic.AnthropicError, pydantic.ValidationError) as exc:
            msg = f"{prompt_name}: {type(exc).__name__}"
            raise LLMError(msg) from exc
        usage = LLMUsage(
            call=prompt_name,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        record_usage(usage)
        _log.info(
            "llm_call",
            call=prompt_name,
            model=message.model,
            duration_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            stop_reason=message.stop_reason,
        )
        if message.stop_reason == "refusal":
            msg = f"{prompt_name}: refused"
            raise LLMRefusalError(msg)
        parsed = message.parsed_output
        if parsed is None:
            msg = f"{prompt_name}: no structured output"
            raise LLMError(msg)
        return parsed


def create_llm_client(
    settings: Settings, prompts: PromptLibrary
) -> tuple[LLMClient, AsyncAnthropic | None]:
    """The configured client, plus the SDK client to close at shutdown (None when disabled)."""
    key = settings.anthropic_api_key.get_secret_value()
    if not key:
        _log.info("llm_disabled")
        return DisabledLLMClient(), None
    sdk = AsyncAnthropic(
        api_key=key, timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries
    )
    client = AnthropicLLMClient(
        sdk, prompts, call_configs(settings), refusal_fallback=settings.llm_refusal_fallback
    )
    return client, sdk
