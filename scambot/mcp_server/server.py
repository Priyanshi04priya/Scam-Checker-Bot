"""Checker tool server (MCP over Streamable HTTP).

Run with ``python -m scambot.mcp_server.server``.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

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
from scambot.core.media_store import AUDIO_FILE, LocalMediaStore
from scambot.core.settings import Settings, get_settings
from scambot.factcheck_index.embed import Embedder, create_embedder
from scambot.factcheck_index.search import PgFactcheckIndex
from scambot.mcp_server.checkers.caption import CaptionChecker
from scambot.mcp_server.checkers.deepfake import DeepfakeChecker
from scambot.mcp_server.checkers.factcheck import ClaimSearch, FactCheckChecker, FactCheckSources
from scambot.mcp_server.checkers.provenance import ProvenanceChecker
from scambot.mcp_server.checkers.scam import ScamChecker
from scambot.mcp_server.checkers.voice_clone import VoiceCloneChecker
from scambot.mcp_server.clients.factcheck_tools import FactCheckToolsClient
from scambot.mcp_server.clients.pagefetch import SafePageFetcher
from scambot.mcp_server.clients.safebrowsing import SafeBrowsingClient, UrlReputation
from scambot.mcp_server.clients.sightengine import ImageAnalyzer, SightengineClient
from scambot.mcp_server.clients.vision import VisionClient, WebSearch
from scambot.mcp_server.clients.voice_spoof import create_voice_detector
from scambot.mcp_server.media import MediaFiles
from scambot.mcp_server.tooling import checker_tool
from scambot.preprocess.stt import silero_vad

SERVER_NAME = "scambot-checkers"
MCP_PATH = "/mcp"

_log = get_logger(component="mcp_server")


class Closeable(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class CheckerDeps:
    scam: ScamChecker
    factcheck: FactCheckChecker
    provenance: ProvenanceChecker
    caption: CaptionChecker
    deepfake: DeepfakeChecker
    voice: VoiceCloneChecker


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

    @server.tool(name="check_provenance")
    @checker_tool("check_provenance")
    async def check_provenance(
        media_key: str,
        trace_id: str,
        reference_date: dt.date | None = None,
        sha256: str | None = None,
        frames: list[str] | None = None,
    ) -> CheckerResult:
        """Find the oldest credible web page that shows this picture."""
        return await deps.provenance.check(media_key, reference_date, frames)

    @server.tool(name="check_caption_context")
    @checker_tool("check_caption_context")
    async def check_caption_context(  # noqa: PLR0913, PLR0917
        trace_id: str,
        caption: str = "",
        claimed_date: dt.date | None = None,
        claimed_place: str | None = None,
        received_on: dt.date | None = None,
        origin: dict[str, Any] | None = None,
    ) -> CheckerResult:
        """Compare the message's claim with the picture's original published context."""
        return await deps.caption.check(caption, claimed_date, claimed_place, origin, received_on)

    @server.tool(name="detect_deepfake")
    @checker_tool("detect_deepfake")
    async def detect_deepfake(
        media_key: str, frames: list[str], trace_id: str, sha256: str | None = None
    ) -> CheckerResult:
        """Rate how likely the picture is AI-generated or altered."""
        return await deps.deepfake.check(media_key, frames)

    @server.tool(name="detect_voice_clone")
    @checker_tool("detect_voice_clone")
    async def detect_voice_clone(
        media_key: str, trace_id: str, audio: str = AUDIO_FILE, sha256: str | None = None
    ) -> CheckerResult:
        """Rate how likely the voice is synthetic or cloned."""
        return await deps.voice.check(media_key, audio)

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


def _scam(
    settings: Settings, config: AppConfig, llm: LLMClient, closeables: list[ServerCloseable]
) -> ScamChecker:
    reputation: UrlReputation | None = None
    if settings.google_api_key.get_secret_value():
        http = httpx.AsyncClient(timeout=config.thresholds.scam.safe_browsing_timeout_s)
        closeables.append(http)
        reputation = SafeBrowsingClient(http, settings.google_api_key)
    else:
        _log.info("safe_browsing_disabled")
    return ScamChecker(config.scam_patterns, config.thresholds.scam, llm, reputation)


def _provenance(
    settings: Settings, config: AppConfig, media: MediaFiles, closeables: list[ServerCloseable]
) -> ProvenanceChecker:
    limits = config.thresholds.provenance
    search: WebSearch | None = None
    if settings.google_api_key.get_secret_value():
        vision_http = httpx.AsyncClient(timeout=limits.vision_timeout_s)
        closeables.append(vision_http)
        search = VisionClient(vision_http, settings.google_api_key)
    else:
        _log.info("vision_disabled")
    fetch_http = httpx.AsyncClient(timeout=limits.fetch.timeout_s, trust_env=False)
    closeables.append(fetch_http)
    fetcher = SafePageFetcher(fetch_http, limits.fetch)
    return ProvenanceChecker(search, fetcher, media, limits, config.sources)


def _deepfake(
    settings: Settings, config: AppConfig, media: MediaFiles, closeables: list[ServerCloseable]
) -> DeepfakeChecker:
    analyzer: ImageAnalyzer | None = None
    if settings.sightengine_user and settings.sightengine_secret.get_secret_value():
        http = httpx.AsyncClient(timeout=config.thresholds.deepfake.timeout_s)
        closeables.append(http)
        analyzer = SightengineClient(http, settings.sightengine_user, settings.sightengine_secret)
    else:
        _log.info("deepfake_disabled")
    return DeepfakeChecker(analyzer, media, config.thresholds.deepfake)


def _voice(settings: Settings, config: AppConfig, media: MediaFiles) -> VoiceCloneChecker:
    cfg = config.thresholds.voice
    detector = create_voice_detector(settings, cfg)
    return VoiceCloneChecker(detector, media, cfg, silero_vad(cfg.min_silence_ms))


def warm_up_voice(checker: VoiceCloneChecker) -> VoiceCloneChecker:
    """Load the voice model now so the first request does not wait; drop it if it cannot load."""
    detector = checker.detector
    if detector is None:
        return checker
    try:
        detector.warm_up()
    except Exception as exc:  # a missing or broken model only switches the voice check off
        _log.warning("voice_detector_unavailable", reason="load_failed", error=str(exc)[:40])
        return checker.without_detector()
    return checker


def create_deps(settings: Settings, config: AppConfig) -> tuple[CheckerDeps, list[ServerCloseable]]:
    """Checker dependencies plus the clients that hold connections (kept for the process)."""
    closeables: list[ServerCloseable] = []
    llm, anthropic_client = create_llm_client(settings, load_prompts())
    if anthropic_client is not None:
        closeables.append(anthropic_client)
    media = MediaFiles(LocalMediaStore(settings.media_dir))
    deps = CheckerDeps(
        scam=_scam(settings, config, llm, closeables),
        factcheck=_factcheck(settings, config, llm, closeables),
        provenance=_provenance(settings, config, media, closeables),
        caption=CaptionChecker(llm, config.thresholds.caption, config.thresholds.provenance),
        deepfake=_deepfake(settings, config, media, closeables),
        voice=_voice(settings, config, media),
    )
    return deps, closeables


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and prompts (an error stops the start), then serve."""
    settings = get_settings()
    configure_logging(settings.log_level)
    config = load_config()
    deps, _closeables = create_deps(settings, config)
    deps = replace(deps, voice=warm_up_voice(deps.voice))
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
