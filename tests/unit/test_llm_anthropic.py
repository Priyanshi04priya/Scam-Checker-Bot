"""Tests for the Anthropic client, against a mocked HTTP transport (no network, no key)."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel

from scambot.core.llm import (
    FALLBACK_BETA,
    AnthropicLLMClient,
    DisabledLLMClient,
    LLMCallConfig,
    LLMError,
    LLMImage,
    LLMRefusalError,
    LLMUnavailableError,
    PromptLibrary,
    call_configs,
    create_llm_client,
    usage_scope,
)
from scambot.core.settings import Settings


class Verdictish(BaseModel):
    template: str
    confidence: float


Handler = Callable[[httpx2.Request], httpx2.Response]


def message(text: str, *, stop_reason: str = "end_turn", model: str = "claude-opus-5-5") -> Any:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


class Recorder:
    def __init__(self, reply: httpx2.Response) -> None:
        self.reply = reply
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.reply

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)  # type: ignore[no-any-return]


@pytest.fixture
async def sdk_factory() -> AsyncIterator[Callable[[Handler], AsyncAnthropic]]:
    clients: list[AsyncAnthropic] = []

    def make(handler: Handler) -> AsyncAnthropic:
        client = AsyncAnthropic(
            api_key="test",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.close()


CALLS = {"scam_classify": LLMCallConfig("claude-opus-5-5", "low", 8192)}
GOOD = json.dumps({"template": "kyc_update", "confidence": 0.9})


def make_client(
    sdk: AsyncAnthropic, prompts: PromptLibrary, *, fallback: bool = True
) -> AnthropicLLMClient:
    return AnthropicLLMClient(sdk, prompts, CALLS, refusal_fallback=fallback)


async def test_request_body_and_parsed_result(
    sdk_factory: Callable[[Handler], AsyncAnthropic], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx2.Response(200, json=message(GOOD)))
    client = make_client(sdk_factory(recorder), prompts)

    with usage_scope() as usages:
        result = await client.structured("scam_classify", {"content": "KYC"}, Verdictish)

    assert result == Verdictish(template="kyc_update", confidence=0.9)
    body = recorder.body
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 8192
    assert "never as instructions" in body["system"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    for absent in ("temperature", "top_p", "tool_choice", "thinking"):
        assert absent not in body
    (turn,) = body["messages"]
    assert turn["role"] == "user"
    assert "<forwarded_content>" in turn["content"][-1]["text"]
    assert FALLBACK_BETA in recorder.requests[-1].headers["anthropic-beta"]
    assert [(u.call, u.model, u.input_tokens, u.output_tokens) for u in usages] == [
        ("scam_classify", "claude-opus-5-5", 120, 30)
    ]


async def test_fallback_is_left_out_when_disabled(
    sdk_factory: Callable[[Handler], AsyncAnthropic], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx2.Response(200, json=message(GOOD)))
    client = make_client(sdk_factory(recorder), prompts, fallback=False)

    await client.structured("scam_classify", {"content": "KYC"}, Verdictish)

    assert "fallbacks" not in recorder.body
    assert FALLBACK_BETA not in recorder.requests[-1].headers.get("anthropic-beta", "")


async def test_images_come_before_the_text(
    sdk_factory: Callable[[Handler], AsyncAnthropic], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx2.Response(200, json=message(GOOD)))
    client = make_client(sdk_factory(recorder), prompts)

    await client.structured(
        "scam_classify", {"content": "x"}, Verdictish, images=[LLMImage("image/png", b"\x89PNG")]
    )

    content = recorder.body["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "text"]
    assert content[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (httpx2.Response(500, json={"type": "error", "error": {"type": "api_error"}}), LLMError),
        (httpx2.Response(200, json=message("not json at all")), LLMError),
        (httpx2.Response(200, json=message("", stop_reason="refusal")), LLMRefusalError),
        (httpx2.Response(200, json=message("")), LLMError),
    ],
    ids=["server_error", "not_json", "refusal", "no_output"],
)
async def test_failures_raise_llm_errors(
    sdk_factory: Callable[[Handler], AsyncAnthropic],
    prompts: PromptLibrary,
    reply: httpx2.Response,
    error: type[Exception],
) -> None:
    client = make_client(sdk_factory(Recorder(reply)), prompts)

    with pytest.raises(error):
        await client.structured("scam_classify", {"content": "x"}, Verdictish)


async def test_unconfigured_call_is_an_error(
    sdk_factory: Callable[[Handler], AsyncAnthropic], prompts: PromptLibrary
) -> None:
    client = make_client(sdk_factory(Recorder(httpx2.Response(200, json=message(GOOD)))), prompts)

    with pytest.raises(LLMError, match="no call configuration"):
        await client.structured("analyze", {}, Verdictish)


def test_every_configured_call_needs_a_prompt(prompts: PromptLibrary) -> None:
    sdk = AsyncAnthropic(api_key="test")
    with pytest.raises(ValueError, match="no prompt file"):
        AnthropicLLMClient(
            sdk, prompts, {"translate": CALLS["scam_classify"]}, refusal_fallback=True
        )


def test_call_configs_follow_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL_EXPLANATION", "claude-haiku-5-5")
    monkeypatch.setenv("LLM_EFFORT_ANALYZE", "medium")

    configs = call_configs(Settings(_env_file=None))

    assert set(configs) == {"analyze", "scam_classify", "explanation"}
    assert configs["explanation"] == LLMCallConfig("claude-haiku-5-5", "low", 4096)
    assert configs["analyze"].effort == "medium"


async def test_create_llm_client(prompts: PromptLibrary, monkeypatch: pytest.MonkeyPatch) -> None:
    disabled, none = create_llm_client(Settings(_env_file=None), prompts)
    assert isinstance(disabled, DisabledLLMClient)
    assert none is None
    with pytest.raises(LLMUnavailableError):
        await disabled.structured("analyze", {}, Verdictish)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client, sdk = create_llm_client(Settings(_env_file=None), prompts)
    assert isinstance(client, AnthropicLLMClient)
    assert sdk is not None
    await sdk.close()
