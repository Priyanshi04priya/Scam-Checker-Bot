"""Fact-check text end to end: intake, queue, worker, in-process checker server with the real
fact-check and scam checkers over an in-memory index."""

import io
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

import pytest
from arq.connections import ArqRedis
from mcp.server.mcpserver import MCPServer

from scambot.agent.analyze import Analysis
from scambot.agent.executor import McpToolCaller
from scambot.core.config import AppConfig
from scambot.core.llm import LLMError
from scambot.core.locales import Locales
from scambot.core.logging import configure_logging
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from scambot.mcp_server.checkers.factcheck import FactCheckChecker, FactCheckSources
from scambot.mcp_server.server import CheckerDeps, build_server
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    FailingIndex,
    FakeAdapter,
    FakeLLMClient,
    InMemoryStore,
    KeywordEmbedder,
    fake_update,
    judge,
    seed,
)
from tests.integration.test_scam_text import check
from tests.samples import (
    LAPTOP_CLAIM_HI,
    LAPTOP_FACTCHECK,
    LAPTOP_FACTCHECK_URL,
    LAPTOP_FORWARD_HI,
    UNRELATED_CLAIM_EN,
    UNRELATED_FACTCHECKS,
)


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


@pytest.fixture(autouse=True)
async def _seed(factcheck_index: InMemoryFactcheckIndex, embedder: KeywordEmbedder) -> None:
    await seed(factcheck_index, embedder, LAPTOP_FACTCHECK, *UNRELATED_FACTCHECKS)


def analysis_of(claim: str, language: str) -> Any:
    def answer(_variables: Mapping[str, str]) -> Analysis:
        return Analysis(
            claims=[claim],
            reply_language=language,
            claimed_date=None,
            claimed_place=None,
            extra_checkers=[],
            reason="",
        )

    return answer


Send = Callable[..., Awaitable[str]]


@pytest.fixture
def send(
    redis: ArqRedis, adapter: FakeAdapter, locales: Locales, pipeline_deps: PipelineDepsFactory
) -> Send:
    """Send one message through intake and the worker; returns the final reply."""

    async def run(update: Mapping[str, Any], caller: McpToolCaller | None = None) -> str:
        return await check(redis, adapter, locales, pipeline_deps(caller=caller), update)

    return run


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Callable[[], str]:
    """Everything logged during the test, from the standard logger and from structlog."""
    caplog.set_level(logging.DEBUG)
    stream = io.StringIO()
    configure_logging("DEBUG", stream)
    return lambda: caplog.text + stream.getvalue()


async def test_hindi_claim_is_answered_with_the_english_fact_check(
    send: Send,
    locales: Locales,
    fake_llm: FakeLLMClient,
    store: InMemoryStore,
    logs: Callable[[], str],
) -> None:
    fake_llm.responses["analyze"] = analysis_of(LAPTOP_CLAIM_HI, "hi")
    fake_llm.responses["factcheck_relevance"] = judge()
    fake_llm.responses["explanation"] = LLMError("down")

    reply = await send(fake_update(1, LAPTOP_FORWARD_HI, lang="en"))

    hi = locales.get("hi")
    assert reply.startswith("🔴 उच्च जोखिम · झूठा/भ्रामक")
    assert hi.verdict.explanations.false_misleading in reply
    assert f"{hi.verdict.sources}:" in reply
    assert LAPTOP_FACTCHECK_URL in reply
    assert reply.endswith(hi.disclaimer)
    (row,) = store.requests.values()
    assert {c.result.checker for c in row.checks} == {"analyze_scam_patterns", "fact_check_claim"}
    assert row.verdict is not None
    assert row.verdict.deciding_checker == "fact_check_claim"
    logged = logs()
    assert "job_replied" in logged
    assert LAPTOP_CLAIM_HI not in logged
    assert "लैपटॉप" not in logged


async def test_unrelated_claim_has_no_match(
    send: Send, fake_llm: FakeLLMClient, logs: Callable[[], str]
) -> None:
    fake_llm.responses["analyze"] = analysis_of(UNRELATED_CLAIM_EN, "en")
    fake_llm.responses["factcheck_relevance"] = judge(about="taj mahal")

    reply = await send(fake_update(2, UNRELATED_CLAIM_EN))

    assert reply.startswith("🟢 Low risk · No issues found")
    assert "factly.in" not in reply
    assert "Taj Mahal" not in logs()


async def test_fact_check_failure_names_the_missing_check(
    send: Send,
    locales: Locales,
    fake_llm: FakeLLMClient,
    app_config: AppConfig,
    checker_deps: CheckerDeps,
) -> None:
    fake_llm.responses["analyze"] = analysis_of(UNRELATED_CLAIM_EN, "en")
    broken = FactCheckChecker(
        FactCheckSources(api=None, index=FailingIndex(), embedder=KeywordEmbedder()),
        fake_llm,
        app_config.thresholds.factcheck,
        app_config.sources,
    )
    server: MCPServer = build_server(replace(checker_deps, factcheck=broken))

    reply = await send(fake_update(3, UNRELATED_CLAIM_EN), McpToolCaller(server, 5))

    en = locales.get("en").verdict
    assert en.unavailable.replace("{checks}", en.checkers.fact_check_claim) in reply
