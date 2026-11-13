"""Shared pytest fixtures."""

from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import structlog
from arq.connections import ArqRedis
from fakeredis import FakeAsyncRedis, FakeServer
from mcp.server.mcpserver import MCPServer

from scambot.agent.executor import Executor, McpToolCaller, ToolCaller
from scambot.core.config import AppConfig, load_config
from scambot.core.llm import LLMClient, PromptLibrary, load_prompts
from scambot.core.locales import Locales, load_locales
from scambot.core.media_store import LocalMediaStore, MediaStore
from scambot.core.settings import Settings, get_settings
from scambot.core.store import RequestStore
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from scambot.mcp_server.checkers.caption import CaptionChecker
from scambot.mcp_server.checkers.deepfake import DeepfakeChecker
from scambot.mcp_server.checkers.factcheck import FactCheckChecker, FactCheckSources
from scambot.mcp_server.checkers.provenance import ProvenanceChecker
from scambot.mcp_server.checkers.scam import ScamChecker
from scambot.mcp_server.media import MediaFiles
from scambot.mcp_server.server import CheckerDeps, build_server
from scambot.preprocess.entities import NoopExpander
from scambot.preprocess.language import ScriptLanguageDetector
from scambot.preprocess.media import MediaEngines
from scambot.worker.pipeline import PipelineDeps
from tests.fakes import (
    FakeAdapter,
    FakeImageAnalyzer,
    FakeKeyframes,
    FakeLLMClient,
    FakeOcr,
    FakePageFetcher,
    FakeStt,
    FakeWebSearch,
    InMemoryStore,
    KeywordEmbedder,
)
from tests.samples import EMBEDDING_ALIASES


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every settings variable from the environment, stop the project ``.env`` from
    being read and clear the settings cache."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Restore structlog's default configuration after each test."""
    yield
    structlog.reset_defaults()


@pytest.fixture
async def redis() -> AsyncIterator[ArqRedis]:
    """A queue-capable Redis client backed by a private in-memory server."""
    client = ArqRedis(connection_pool=FakeAsyncRedis(server=FakeServer()).connection_pool)
    yield client
    # The pool was passed in explicitly, so the client would otherwise leave it open.
    await client.aclose(close_connection_pool=True)


@pytest.fixture(scope="session")
def locales() -> Locales:
    return load_locales()


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return load_config()


@pytest.fixture(scope="session")
def prompts() -> PromptLibrary:
    return load_prompts()


@pytest.fixture
def fake_llm(prompts: PromptLibrary) -> FakeLLMClient:
    """An LLM with no canned answers: every call fails until a test adds responses."""
    return FakeLLMClient(prompts)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def scam_checker(app_config: AppConfig, fake_llm: FakeLLMClient) -> ScamChecker:
    return ScamChecker(app_config.scam_patterns, app_config.thresholds.scam, fake_llm, None)


@pytest.fixture
def factcheck_index() -> InMemoryFactcheckIndex:
    return InMemoryFactcheckIndex()


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder(EMBEDDING_ALIASES)


@pytest.fixture
def factcheck_checker(
    app_config: AppConfig,
    fake_llm: FakeLLMClient,
    factcheck_index: InMemoryFactcheckIndex,
    embedder: KeywordEmbedder,
) -> FactCheckChecker:
    return FactCheckChecker(
        FactCheckSources(api=None, index=factcheck_index, embedder=embedder),
        fake_llm,
        app_config.thresholds.factcheck,
        app_config.sources,
    )


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    return tmp_path / "media"


@pytest.fixture
def media_store(media_root: Path) -> MediaStore:
    return LocalMediaStore(media_root)


@pytest.fixture
def web_search() -> FakeWebSearch:
    return FakeWebSearch()


@pytest.fixture
def analyzer() -> FakeImageAnalyzer:
    return FakeImageAnalyzer()


@pytest.fixture
def page_fetcher() -> FakePageFetcher:
    return FakePageFetcher()


@pytest.fixture
def provenance_checker(
    app_config: AppConfig,
    web_search: FakeWebSearch,
    page_fetcher: FakePageFetcher,
    media_store: MediaStore,
) -> ProvenanceChecker:
    return ProvenanceChecker(
        web_search,
        page_fetcher,
        MediaFiles(media_store),
        app_config.thresholds.provenance,
        app_config.sources,
    )


@pytest.fixture
def caption_checker(app_config: AppConfig, fake_llm: FakeLLMClient) -> CaptionChecker:
    return CaptionChecker(fake_llm, app_config.thresholds.caption, app_config.thresholds.provenance)


@pytest.fixture
def deepfake_checker(
    app_config: AppConfig, analyzer: FakeImageAnalyzer, media_store: MediaStore
) -> DeepfakeChecker:
    return DeepfakeChecker(analyzer, MediaFiles(media_store), app_config.thresholds.deepfake)


@pytest.fixture
def checker_deps(
    scam_checker: ScamChecker,
    factcheck_checker: FactCheckChecker,
    provenance_checker: ProvenanceChecker,
    caption_checker: CaptionChecker,
    deepfake_checker: DeepfakeChecker,
) -> CheckerDeps:
    return CheckerDeps(
        scam=scam_checker,
        factcheck=factcheck_checker,
        provenance=provenance_checker,
        caption=caption_checker,
        deepfake=deepfake_checker,
    )


@pytest.fixture
def mcp_server(checker_deps: CheckerDeps) -> MCPServer:
    return build_server(checker_deps)


@pytest.fixture
def media_engines(app_config: AppConfig) -> Iterator[MediaEngines]:
    """Fake engines that read nothing; tests replace the ones they need."""
    pool = ThreadPoolExecutor(max_workers=2)
    yield MediaEngines(
        ocr=FakeOcr(),
        stt=FakeStt(),
        keyframes=FakeKeyframes(),
        language=ScriptLanguageDetector(),
        pool=pool,
        limits=app_config.thresholds.media,
    )
    pool.shutdown(wait=True)


PipelineDepsFactory = Callable[..., PipelineDeps]


@pytest.fixture
def pipeline_deps(
    adapter: FakeAdapter,
    locales: Locales,
    app_config: AppConfig,
    fake_llm: FakeLLMClient,
    media_store: MediaStore,
    media_engines: MediaEngines,
    request: pytest.FixtureRequest,
) -> PipelineDepsFactory:
    """Build pipeline dependencies from the fixtures, with any of them replaced by keyword."""
    store: InMemoryStore = request.getfixturevalue("store")
    mcp_server: MCPServer = request.getfixturevalue("mcp_server")

    def build(
        *,
        llm: LLMClient | None = None,
        caller: ToolCaller | None = None,
        request_store: RequestStore | None = None,
        config: AppConfig | None = None,
        engines: MediaEngines | None = None,
        store_override: MediaStore | None = None,
    ) -> PipelineDeps:
        return PipelineDeps(
            adapters={"telegram": adapter},
            locales=locales,
            config=config or app_config,
            llm=llm or fake_llm,
            executor=Executor(caller or McpToolCaller(mcp_server, 5)),
            store=request_store or store,
            expander=NoopExpander(),
            media_store=store_override or media_store,
            media=engines or media_engines,
        )

    return build
