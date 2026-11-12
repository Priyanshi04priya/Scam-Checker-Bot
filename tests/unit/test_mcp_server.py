"""Tests for the checker tool server, called in-process (no network)."""

from typing import Any

import httpx
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from sqlalchemy.ext.asyncio import AsyncEngine

from scambot.core.config import AppConfig
from scambot.core.contracts import CheckerResult, Entities
from scambot.core.settings import Settings
from scambot.core.tracing import get_trace_id
from scambot.factcheck_index.embed import SentenceTransformerEmbedder
from scambot.factcheck_index.search import PgFactcheckIndex
from scambot.mcp_server import server as server_module
from scambot.mcp_server.checkers.factcheck import FactCheckChecker
from scambot.mcp_server.checkers.scam import ScamChecker
from scambot.mcp_server.clients.factcheck_tools import FactCheckToolsClient
from scambot.mcp_server.clients.safebrowsing import SafeBrowsingClient
from scambot.mcp_server.server import CheckerDeps, build_server, create_deps, main
from scambot.mcp_server.tooling import checker_tool
from tests.fakes import StubEncoder
from tests.samples import ELECTRICITY_EN


async def test_tools_are_listed_with_their_contracts(mcp_server: MCPServer) -> None:
    async with Client(mcp_server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"analyze_scam_patterns", "fact_check_claim"}
    scam = tools["analyze_scam_patterns"]
    assert set(scam.input_schema["required"]) == {"text", "trace_id"}
    assert {"entities", "language"} <= set(scam.input_schema["properties"])
    assert scam.output_schema is not None
    factcheck = tools["fact_check_claim"]
    assert set(factcheck.input_schema["required"]) == {"claims", "trace_id"}
    assert factcheck.output_schema is not None


async def test_fact_check_output_validates(mcp_server: MCPServer) -> None:
    async with Client(mcp_server) as client:
        reply = await client.call_tool("fact_check_claim", {"claims": [], "trace_id": "t-1"})

    assert not reply.is_error
    result = CheckerResult.model_validate(reply.structured_content)
    assert (result.checker, result.status) == ("fact_check_claim", "inconclusive")


async def test_output_validates_as_a_checker_result(mcp_server: MCPServer) -> None:
    arguments = {"text": ELECTRICITY_EN, "trace_id": "t-1", "entities": {"phones": ["+91"]}}
    async with Client(mcp_server) as client:
        reply = await client.call_tool("analyze_scam_patterns", arguments)

    assert not reply.is_error
    result = CheckerResult.model_validate(reply.structured_content)
    assert (result.checker, result.signal) == ("analyze_scam_patterns", "scam")
    assert result.duration_ms >= 0


class ExplodingChecker(ScamChecker):
    async def check(
        self, text: str, entities: Entities | None, language: str | None
    ) -> CheckerResult:
        msg = "secret detail that must not leak"
        raise RuntimeError(msg)


async def test_checker_exception_becomes_an_error_result(
    app_config: AppConfig, factcheck_checker: FactCheckChecker
) -> None:
    exploding = ExplodingChecker(
        app_config.scam_patterns, app_config.thresholds.scam, _NoLLM(), None
    )
    server = build_server(CheckerDeps(scam=exploding, factcheck=factcheck_checker))

    async with Client(server) as client:
        reply = await client.call_tool("analyze_scam_patterns", {"text": "x", "trace_id": "t"})

    result = CheckerResult.model_validate(reply.structured_content)
    assert (result.status, result.error) == ("error", "RuntimeError")
    assert "secret" not in str(reply.structured_content)


async def test_invalid_arguments_are_a_tool_error(mcp_server: MCPServer) -> None:
    async with Client(mcp_server) as client:
        reply = await client.call_tool("analyze_scam_patterns", {"text": 5})

    assert reply.is_error


async def test_wrapper_uses_the_trace_id() -> None:
    seen: list[str | None] = []

    @checker_tool("analyze_scam_patterns")
    async def tool(trace_id: str) -> CheckerResult:
        seen.append(get_trace_id())
        return CheckerResult(
            checker="analyze_scam_patterns", status="ok", signal="none", confidence=1, duration_ms=0
        )

    await tool(trace_id="trace-42")

    assert seen == ["trace-42"]


class _NoLLM:
    async def structured(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError  # pragma: no cover


def test_main_runs_streamable_http(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(self: MCPServer, transport: str, **kwargs: Any) -> None:
        calls.append((transport, kwargs))

    monkeypatch.setattr(MCPServer, "run", fake_run)
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp:*,localhost:*")
    monkeypatch.setenv("MCP_PORT", "8123")

    assert main() == 0

    ((transport, kwargs),) = calls
    assert transport == "streamable-http"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123
    assert kwargs["streamable_http_path"] == "/mcp"
    assert kwargs["stateless_http"] is True
    assert kwargs["json_response"] is True
    security = kwargs["transport_security"]
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["mcp:*", "localhost:*"]


def test_config_errors_fail_main(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        msg = "invalid config file routing.yaml: boom"
        raise ValueError(msg)

    monkeypatch.setattr(server_module, "load_config", broken)
    monkeypatch.setattr(MCPServer, "run", lambda *_a, **_k: pytest.fail("must not run"))

    with pytest.raises(ValueError, match=r"routing\.yaml"):
        main()


async def _close(closeables: list[Any]) -> None:
    for client in closeables:
        if isinstance(client, httpx.AsyncClient):
            await client.aclose()
        elif isinstance(client, AsyncEngine):
            await client.dispose()
        else:
            await client.close()


async def test_create_deps(app_config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "create_embedder", lambda _settings: None)
    deps, closeables = create_deps(Settings(_env_file=None), app_config)
    assert isinstance(deps.scam, ScamChecker)
    assert [type(c) for c in closeables] == [AsyncEngine]
    sources = deps.factcheck._sources
    assert (sources.api, sources.index, sources.embedder) == (None, None, None)
    await _close(closeables)

    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    deps, closeables = create_deps(Settings(_env_file=None), app_config)
    assert len(closeables) == 4  # LLM, Safe Browsing, database engine, Fact Check Tools
    assert isinstance(deps.scam._reputation, SafeBrowsingClient)
    assert isinstance(deps.factcheck._sources.api, FactCheckToolsClient)
    await _close(closeables)


async def test_create_deps_with_an_embedder(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = SentenceTransformerEmbedder("m", loader=lambda _name: StubEncoder())
    monkeypatch.setattr(server_module, "create_embedder", lambda _settings: loaded)
    deps, closeables = create_deps(Settings(_env_file=None), app_config)
    assert deps.factcheck._sources.embedder is loaded
    assert isinstance(deps.factcheck._sources.index, PgFactcheckIndex)
    await _close(closeables)

    def broken(_name: str) -> StubEncoder:
        msg = "model download failed"
        raise OSError(msg)

    failing = SentenceTransformerEmbedder("m", loader=broken)
    monkeypatch.setattr(server_module, "create_embedder", lambda _settings: failing)
    deps, closeables = create_deps(Settings(_env_file=None), app_config)
    assert deps.factcheck._sources.embedder is None
    assert deps.factcheck._sources.index is None
    await _close(closeables)
