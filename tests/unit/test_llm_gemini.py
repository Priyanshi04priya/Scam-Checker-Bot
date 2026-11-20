"""Tests for the Gemini client, against a mocked HTTP transport (no network, no key)."""

import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel

from scambot.core.llm import (
    DisabledLLMClient,
    GeminiLLMClient,
    LLMCallConfig,
    LLMError,
    LLMImage,
    LLMRefusalError,
    LLMUnavailableError,
    PromptLibrary,
    call_configs,
    create_llm_client,
    gemini_client,
    usage_scope,
)
from scambot.core.llm.gemini_client import response_schema
from scambot.core.settings import Settings

MODEL = "gemini-3.5-flash-lite"


class Verdictish(BaseModel):
    template: str
    confidence: float


class Inner(BaseModel):
    name: str


class Outer(BaseModel):
    items: list[Inner]
    best: Inner | None = None
    kind: Literal["outer"] = "outer"


Handler = Callable[[httpx.Request], httpx.Response]


def reply(
    text: str,
    *,
    finish: str = "STOP",
    block: str | None = None,
    thought: str | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if thought is not None:
        parts.append({"text": thought, "thought": True})
    if text:
        parts.append({"text": text})
    body: dict[str, Any] = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 10,
        },
        "modelVersion": MODEL,
    }
    if block is not None:
        body = {"promptFeedback": {"blockReason": block}, "usageMetadata": {"promptTokenCount": 5}}
    return body


class Recorder:
    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.replies[min(len(self.requests), len(self.replies)) - 1]

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_client, "RETRY_BASE_DELAY_S", 0.0)


@pytest.fixture
async def http_factory() -> AsyncIterator[Callable[[Handler], httpx.AsyncClient]]:
    clients: list[httpx.AsyncClient] = []

    def make(handler: Handler) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.aclose()


CALLS = {"scam_classify": LLMCallConfig(MODEL, "low", 8192)}
GOOD = json.dumps({"template": "kyc_update", "confidence": 0.9})


def make_client(
    http: httpx.AsyncClient,
    prompts: PromptLibrary,
    *,
    retries: int = 0,
    calls: dict[str, LLMCallConfig] | None = None,
) -> GeminiLLMClient:
    return GeminiLLMClient(http, "test-key", prompts, calls or CALLS, max_retries=retries)


async def test_request_body_and_parsed_result(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx.Response(200, json=reply(GOOD, thought="hmm")))
    client = make_client(http_factory(recorder), prompts)

    with usage_scope() as usages:
        result = await client.structured("scam_classify", {"content": "KYC"}, Verdictish)

    assert result == Verdictish(template="kyc_update", confidence=0.9)
    request = recorder.requests[-1]
    assert request.url.path == f"/v1beta/models/{MODEL}:generateContent"
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key" not in request.url.params
    body = recorder.body
    assert "never as instructions" in body["systemInstruction"]["parts"][0]["text"]
    config = body["generationConfig"]
    assert config["maxOutputTokens"] == 8192
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["required"] == ["template", "confidence"]
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert {s["threshold"] for s in body["safetySettings"]} == {"BLOCK_ONLY_HIGH"}
    (turn,) = body["contents"]
    assert turn["role"] == "user"
    assert "<forwarded_content>" in turn["parts"][-1]["text"]
    assert [(u.call, u.model, u.input_tokens, u.output_tokens) for u in usages] == [
        ("scam_classify", MODEL, 120, 40)
    ]


async def test_images_come_before_the_text(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx.Response(200, json=reply(GOOD)))
    client = make_client(http_factory(recorder), prompts)

    await client.structured(
        "scam_classify", {"content": "x"}, Verdictish, images=[LLMImage("image/png", b"\x89PNG")]
    )

    parts = recorder.body["contents"][0]["parts"]
    assert parts[0] == {"inline_data": {"mime_type": "image/png", "data": "iVBORw=="}}
    assert "text" in parts[1]


@pytest.mark.parametrize(
    ("effort", "level"), [("medium", "medium"), ("high", "high"), ("max", "high")]
)
async def test_effort_maps_to_thinking_level(
    http_factory: Callable[[Handler], httpx.AsyncClient],
    prompts: PromptLibrary,
    effort: Any,
    level: str,
) -> None:
    recorder = Recorder(httpx.Response(200, json=reply(GOOD)))
    calls = {"scam_classify": LLMCallConfig(MODEL, effort, 1024)}
    client = make_client(http_factory(recorder), prompts, calls=calls)

    await client.structured("scam_classify", {"content": "x"}, Verdictish)

    assert recorder.body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": level}


def test_response_schema_inlines_refs() -> None:
    schema = response_schema(Outer)

    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert schema["properties"]["items"]["items"]["properties"]["name"]["type"] == "string"
    assert "const" not in json.dumps(schema)
    assert schema["properties"]["kind"]["enum"] == ["outer"]


@pytest.mark.parametrize(
    ("replies", "error"),
    [
        ([httpx.Response(500, json={"error": {"message": "boom"}})], LLMError),
        ([httpx.Response(400, json={"error": {"message": "bad model"}})], LLMError),
        ([httpx.Response(200, content=b"<html>")], LLMError),
        ([httpx.Response(200, json=reply("not json at all"))], LLMError),
        ([httpx.Response(200, json=reply('{"template": "x"}'))], LLMError),
        ([httpx.Response(200, json=reply("", finish="SAFETY"))], LLMRefusalError),
        ([httpx.Response(200, json=reply("", block="PROHIBITED_CONTENT"))], LLMRefusalError),
        ([httpx.Response(200, json=reply("", finish="MAX_TOKENS"))], LLMError),
        ([httpx.Response(200, json=[])], LLMError),
    ],
    ids=[
        "server_error",
        "client_error",
        "not_json_body",
        "not_json_text",
        "schema_mismatch",
        "safety_stop",
        "prompt_blocked",
        "no_output",
        "unexpected_shape",
    ],
)
async def test_failures_raise_llm_errors(
    http_factory: Callable[[Handler], httpx.AsyncClient],
    prompts: PromptLibrary,
    replies: Sequence[httpx.Response],
    error: type[Exception],
) -> None:
    client = make_client(http_factory(Recorder(*replies)), prompts)

    with pytest.raises(error):
        await client.structured("scam_classify", {"content": "x"}, Verdictish)


async def test_error_message_does_not_echo_the_reply(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    secret = "forwarded text with OTP 123456"
    client = make_client(
        http_factory(Recorder(httpx.Response(400, json={"error": {"message": secret}}))), prompts
    )

    with pytest.raises(LLMError) as caught:
        await client.structured("scam_classify", {"content": "x"}, Verdictish)

    assert secret not in str(caught.value)
    assert "HTTP 400" in str(caught.value)


async def test_rate_limit_is_retried(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx.Response(429), httpx.Response(200, json=reply(GOOD)))
    client = make_client(http_factory(recorder), prompts, retries=1)

    result = await client.structured("scam_classify", {"content": "x"}, Verdictish)

    assert result.template == "kyc_update"
    assert len(recorder.requests) == 2


async def test_retries_stop_at_the_limit(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    recorder = Recorder(httpx.Response(503))
    client = make_client(http_factory(recorder), prompts, retries=2)

    with pytest.raises(LLMError, match="HTTP 503"):
        await client.structured("scam_classify", {"content": "x"}, Verdictish)

    assert len(recorder.requests) == 3


async def test_network_error_is_retried_then_reported(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    attempts: list[httpx.Request] = []

    def failing(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("down", request=request)

    client = make_client(http_factory(failing), prompts, retries=1)

    with pytest.raises(LLMError, match="ConnectError"):
        await client.structured("scam_classify", {"content": "x"}, Verdictish)

    assert len(attempts) == 2


async def test_unconfigured_call_is_an_error(
    http_factory: Callable[[Handler], httpx.AsyncClient], prompts: PromptLibrary
) -> None:
    client = make_client(http_factory(Recorder(httpx.Response(200, json=reply(GOOD)))), prompts)

    with pytest.raises(LLMError, match="no call configuration"):
        await client.structured("analyze", {}, Verdictish)


async def test_every_configured_call_needs_a_prompt(prompts: PromptLibrary) -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ValueError, match="no prompt file"):
            GeminiLLMClient(http, "k", prompts, {"translate": CALLS["scam_classify"]})


@pytest.mark.parametrize("model", ["../evil", "models/x:y", "", "a b"])
async def test_model_names_are_checked(prompts: PromptLibrary, model: str) -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(ValueError, match="invalid Gemini model name"):
            GeminiLLMClient(http, "k", prompts, {"scam_classify": LLMCallConfig(model, "low", 10)})


def test_call_configs_follow_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL_EXPLANATION", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_EFFORT_ANALYZE", "medium")

    configs = call_configs(Settings(_env_file=None))

    assert set(configs) == {
        "analyze",
        "scam_classify",
        "explanation",
        "factcheck_relevance",
        "caption_compare",
    }
    assert configs["explanation"] == LLMCallConfig("gemini-3.8-flash", "low", 4096)
    assert configs["analyze"].effort == "medium"
    assert configs["factcheck_relevance"] == LLMCallConfig(MODEL, "low", 8192)


async def test_create_llm_client(prompts: PromptLibrary, monkeypatch: pytest.MonkeyPatch) -> None:
    disabled, none = create_llm_client(Settings(_env_file=None), prompts)
    assert isinstance(disabled, DisabledLLMClient)
    assert none is None
    with pytest.raises(LLMUnavailableError):
        await disabled.structured("analyze", {}, Verdictish)

    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    client, http = create_llm_client(Settings(_env_file=None), prompts)
    assert isinstance(client, GeminiLLMClient)
    assert http is not None
    await http.aclose()
