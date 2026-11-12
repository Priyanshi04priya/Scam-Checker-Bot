"""Checker tool server (MCP over Streamable HTTP).

Run with ``python -m scambot.mcp_server.server``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx
from anthropic import AsyncAnthropic
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy.ext.asyncio import AsyncEngine

from scambot.core.config import AppConfig, load_config
from scambot.core.contracts import CheckerResult, Entities
from scambot.core.db import create_engine, create_session_factory
from scambot.core.llm import LLMClient, create_llm_client, load_prompts
from scambot.core.logging import configure_logging, get_logger
from scambot.core.settings import Settings, get_settings
from scambot.factcheck_index.embed import Embedder, create_embedder
from scambot.factcheck_index.search import PgFactcheckIndex
from scambot.mcp_server.checkers.factcheck import ClaimSearch, FactCheckChecker, FactCheckSources
from scambot.mcp_server.checkers.scam import ScamChecker
from scambot.mcp_server.clients.factcheck_tools import FactCheckToolsClient
from scambot.mcp_server.clients.safebrowsing import SafeBrowsingClient, UrlReputation
from scambot.mcp_server.tooling import checker_tool

SERVER_NAME = "scambot-checkers"
MCP_PATH = "/mcp"

_log = get_logger(component="mcp_server")


class Closeable(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class CheckerDeps:
    scam: ScamChecker
    factcheck: FactCheckChecker


def build_server(deps: CheckerDeps) -> MCPServer:
    """The tool server with every implemented checker registered."""
    server = MCPServer(SERVER_NAME)

    @server.tool(name="analyze_scam_patterns")
    @checker_tool("analyze_scam_patterns")
    async def analyze_scam_patterns(
        text: str, trace_id: str, entities: Entities | None = None, language: str | None = None
    ) -> CheckerResult:
        """Look for known scam templates and risk signals in a message."""
        return await deps.scam.check(text, entities, language)

    @server.tool(name="fact_check_claim")
    @checker_tool("fact_check_claim")
    async def fact_check_claim(claims: list[str], trace_id: str) -> CheckerResult:
        """Look for published fact-checks of the message's claims."""
        return await deps.factcheck.check(claims)

    return server


ServerCloseable = httpx.AsyncClient | AsyncAnthropic | AsyncEngine


def _load_embedder(settings: Settings) -> Embedder | None:
    embedder = create_embedder(settings)
    if embedder is None:
        return None
    try:
        embedder.load()
    except Exception as exc:  # a missing or broken model only disables the index search
        _log.warning("embedder_unavailable", reason="load_failed", error=type(exc).__name__)
        return None
    return embedder


def _factcheck(
    settings: Settings, config: AppConfig, llm: LLMClient, closeables: list[ServerCloseable]
) -> FactCheckChecker:
    engine = create_engine(settings.database_url)  # connects only when first used
    closeables.append(engine)
    embedder = _load_embedder(settings)
    index = PgFactcheckIndex(create_session_factory(engine)) if embedder is not None else None
    api: ClaimSearch | None = None
    if settings.google_api_key.get_secret_value():
        http = httpx.AsyncClient(timeout=config.thresholds.factcheck.api_timeout_s)
        closeables.append(http)
        api = FactCheckToolsClient(http, settings.google_api_key)
    else:
        _log.info("factcheck_api_disabled")
    return FactCheckChecker(
        FactCheckSources(api=api, index=index, embedder=embedder),
        llm,
        config.thresholds.factcheck,
        config.sources,
    )


def create_deps(settings: Settings, config: AppConfig) -> tuple[CheckerDeps, list[ServerCloseable]]:
    """Checker dependencies plus the clients that hold connections (kept for the process)."""
    closeables: list[ServerCloseable] = []
    llm, anthropic_client = create_llm_client(settings, load_prompts())
    if anthropic_client is not None:
        closeables.append(anthropic_client)
    reputation: UrlReputation | None = None
    if settings.google_api_key.get_secret_value():
        http = httpx.AsyncClient(timeout=config.thresholds.scam.safe_browsing_timeout_s)
        closeables.append(http)
        reputation = SafeBrowsingClient(http, settings.google_api_key)
    else:
        _log.info("safe_browsing_disabled")
    scam = ScamChecker(config.scam_patterns, config.thresholds.scam, llm, reputation)
    factcheck = _factcheck(settings, config, llm, closeables)
    return CheckerDeps(scam=scam, factcheck=factcheck), closeables


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and prompts (an error stops the start), then serve."""
    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_config()
    deps, _closeables = create_deps(settings, config)
    server = build_server(deps)
    _log.info("mcp_server_starting", host=settings.mcp_host, port=settings.mcp_port)
    server.run(
        "streamable-http",
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=MCP_PATH,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(settings.mcp_allowed_hosts),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
