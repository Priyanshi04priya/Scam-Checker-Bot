"""LLM client interface and implementations."""

from scambot.core.llm.anthropic_client import (
    CALL_NAMES,
    FALLBACK_BETA,
    AnthropicLLMClient,
    call_configs,
    create_llm_client,
)
from scambot.core.llm.base import (
    DisabledLLMClient,
    Effort,
    LLMCallConfig,
    LLMClient,
    LLMError,
    LLMImage,
    LLMRefusalError,
    LLMUnavailableError,
)
from scambot.core.llm.prompts import PROMPTS_DIR, PromptLibrary, load_prompts
from scambot.core.llm.usage import LLMUsage, cost_usd, record_usage, usage_scope

__all__ = [
    "CALL_NAMES",
    "FALLBACK_BETA",
    "PROMPTS_DIR",
    "AnthropicLLMClient",
    "DisabledLLMClient",
    "Effort",
    "LLMCallConfig",
    "LLMClient",
    "LLMError",
    "LLMImage",
    "LLMRefusalError",
    "LLMUnavailableError",
    "LLMUsage",
    "PromptLibrary",
    "call_configs",
    "cost_usd",
    "create_llm_client",
    "load_prompts",
    "record_usage",
    "usage_scope",
]
