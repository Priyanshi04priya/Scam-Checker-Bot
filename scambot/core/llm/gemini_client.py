"""Structured calls to the Google Gemini API (``generateContent``, free tier friendly)."""

import asyncio
import base64
import copy
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx
import pydantic
from pydantic import BaseModel

from scambot.core.llm.base import (
    DisabledLLMClient,
    Effort,
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

CALL_NAMES: Final = (
    "analyze",
    "scam_classify",
    "explanation",
    "factcheck_relevance",
    "caption_compare",
)
GEMINI_API_URL: Final = "https://generativelanguage.googleapis.com"
# Effort setting -> Gemini thinking level.
THINKING_LEVELS: Final[Mapping[Effort, str]] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
# Forwarded scams and hoaxes often look "dangerous" or "harassing" by themselves; only block
# content the model rates as highly harmful, so ordinary scam messages still get checked.
SAFETY_CATEGORIES: Final = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)
REFUSAL_REASONS: Final = frozenset(
    {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)
RETRY_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
RETRY_BASE_DELAY_S: Final = 0.5
_MODEL_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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
        "caption_compare": LLMCallConfig(
            settings.llm_model_caption_compare,
            settings.llm_effort_caption_compare,
            settings.llm_max_tokens_caption_compare,
        ),
    }


def response_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema with every ``$ref`` inlined and ``const`` written as ``enum``."""
    full = schema.model_json_schema()
    defs: dict[str, Any] = full.pop("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name in seen:
                    msg = f"recursive schema {name!r} is not supported"
                    raise LLMError(msg)
                target = copy.deepcopy(defs[name])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**target, **extra}, (*seen, name))
            out = {key: resolve(value, seen) for key, value in node.items()}
            if "const" in out and "enum" not in out:
                # Gemini's schema subset has no ``const``; a one-value enum means the same.
                out["enum"] = [out.pop("const")]
            return out
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        return node

    resolved: dict[str, Any] = resolve(full, ())
    return resolved


def _contents(user: str, images: Sequence[LLMImage] | None) -> list[dict[str, Any]]:
    """The user turn: images first, then the rendered prompt text."""
    parts: list[dict[str, Any]] = [
        {
            "inline_data": {
                "mime_type": image.media_type,
                "data": base64.b64encode(image.data).decode("ascii"),
            }
        }
        for image in images or ()
    ]
    parts.append({"text": user})
    return [{"role": "user", "parts": parts}]


def _text(reply: Mapping[str, Any]) -> str:
    candidate = (reply.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    return "".join(
        part.get("text", "") for part in parts if isinstance(part, dict) and not part.get("thought")
    )


def _usage(prompt_name: str, model: str, reply: Mapping[str, Any]) -> LLMUsage:
    meta = reply.get("usageMetadata") or {}
    output = int(meta.get("candidatesTokenCount") or 0) + int(meta.get("thoughtsTokenCount") or 0)
    return LLMUsage(
        call=prompt_name,
        model=model,
        input_tokens=int(meta.get("promptTokenCount") or 0),
        output_tokens=output,
    )


class GeminiLLMClient:
    """``LLMClient`` backed by Gemini ``generateContent`` with JSON-schema structured output."""

    def __init__(  # noqa: PLR0913
        self,
        http: httpx.AsyncClient,
        api_key: str,
        prompts: PromptLibrary,
        calls: Mapping[str, LLMCallConfig],
        *,
        max_retries: int = 1,
        base_url: str = GEMINI_API_URL,
    ) -> None:
        missing = sorted(set(calls) - prompts.names)
        if missing:
            msg = f"no prompt file for LLM calls: {', '.join(missing)}"
            raise ValueError(msg)
        bad = sorted(c.model for c in calls.values() if not _MODEL_NAME.fullmatch(c.model))
        if bad:
            msg = f"invalid Gemini model name: {', '.join(bad)}"
            raise ValueError(msg)
        self._http = http
        self._api_key = api_key
        self._prompts = prompts
        self._calls = dict(calls)
        self._max_retries = max_retries
        self._base_url = base_url.rstrip("/")

    def _body(
        self,
        config: LLMCallConfig,
        system: str,
        user: str,
        schema: type[BaseModel],
        images: Sequence[LLMImage] | None,
    ) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": _contents(user, images),
            "generationConfig": {
                "maxOutputTokens": config.max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema(schema),
                "thinkingConfig": {"thinkingLevel": THINKING_LEVELS[config.effort]},
            },
            "safetySettings": [
                {"category": category, "threshold": "BLOCK_ONLY_HIGH"}
                for category in SAFETY_CATEGORIES
            ],
        }

    async def _post(self, prompt_name: str, model: str, body: Mapping[str, Any]) -> Any:
        """POST with a few retries on rate limits, server errors and network failures."""
        url = f"{self._base_url}/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": self._api_key}
        attempt = 0
        while True:
            try:
                response = await self._http.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    msg = f"{prompt_name}: {type(exc).__name__}"
                    raise LLMError(msg) from None
            else:
                if response.status_code == httpx.codes.OK:
                    try:
                        return response.json()
                    except ValueError:
                        msg = f"{prompt_name}: reply is not JSON"
                        raise LLMError(msg) from None
                if response.status_code not in RETRY_STATUSES or attempt >= self._max_retries:
                    # Only the status: the error body can echo the forwarded content.
                    msg = f"{prompt_name}: HTTP {response.status_code}"
                    raise LLMError(msg)
            attempt += 1
            await asyncio.sleep(RETRY_BASE_DELAY_S * 2 ** (attempt - 1))

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
        started = time.perf_counter()
        reply = await self._post(
            prompt_name, config.model, self._body(config, system, user, schema, images)
        )
        if not isinstance(reply, dict):
            msg = f"{prompt_name}: unexpected reply"
            raise LLMError(msg)
        usage = _usage(prompt_name, config.model, reply)
        record_usage(usage)
        candidate = (reply.get("candidates") or [{}])[0]
        finish = candidate.get("finishReason")
        blocked = (reply.get("promptFeedback") or {}).get("blockReason")
        _log.info(
            "llm_call",
            call=prompt_name,
            model=config.model,
            duration_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            stop_reason=blocked or finish,
        )
        if blocked or finish in REFUSAL_REASONS:
            msg = f"{prompt_name}: refused"
            raise LLMRefusalError(msg)
        text = _text(reply)
        if not text.strip():
            msg = f"{prompt_name}: no structured output"
            raise LLMError(msg)
        try:
            return schema.model_validate_json(text)
        except pydantic.ValidationError as exc:
            msg = f"{prompt_name}: {type(exc).__name__}"
            raise LLMError(msg) from None


def create_llm_client(
    settings: Settings, prompts: PromptLibrary
) -> tuple[LLMClient, httpx.AsyncClient | None]:
    """The configured client, plus the HTTP client to close at shutdown (None when disabled)."""
    key = settings.gemini_api_key.get_secret_value()
    if not key:
        _log.info("llm_disabled")
        return DisabledLLMClient(), None
    http = httpx.AsyncClient(timeout=settings.llm_timeout_s)
    client = GeminiLLMClient(
        http, key, prompts, call_configs(settings), max_retries=settings.llm_max_retries
    )
    return client, http
