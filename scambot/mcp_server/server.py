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

from scambot.core.config import AppConfig, load_config
from scambot.core.contracts import CheckerResult, Entities
from scambot.core.llm import create_llm_client, load_prompts
from scambot.core.logging import configure_logging, get_logger
from scambot.core.settings import Settings, get_settings
from scambot.mcp_server.checkers.scam import ScamChecker
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

    return server


def create_deps(
    settings: Settings, config: AppConfig
) -> tuple[CheckerDeps, list[httpx.AsyncClient | AsyncAnthropic]]:
    """Checker dependencies plus the clients that hold connections (kept for the process)."""
    closeables: list[httpx.AsyncClient | AsyncAnthropic] = []
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
    return CheckerDeps(scam=scam), closeables


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
