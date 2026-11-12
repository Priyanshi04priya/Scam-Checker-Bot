"""Shared pytest fixtures."""

from collections.abc import AsyncIterator, Callable, Iterator

import pytest
import structlog
from arq.connections import ArqRedis
from fakeredis import FakeAsyncRedis, FakeServer
from mcp.server.mcpserver import MCPServer

from scambot.agent.executor import Executor, McpToolCaller, ToolCaller
from scambot.core.config import AppConfig, load_config
from scambot.core.llm import LLMClient, PromptLibrary, load_prompts
from scambot.core.locales import Locales, load_locales
from scambot.core.settings import Settings, get_settings
from scambot.core.store import RequestStore
from scambot.mcp_server.checkers.scam import ScamChecker
from scambot.mcp_server.server import CheckerDeps, build_server
from scambot.preprocess.entities import NoopExpander
from scambot.worker.pipeline import PipelineDeps
from tests.fakes import FakeAdapter, FakeLLMClient, InMemoryStore


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
def mcp_server(scam_checker: ScamChecker) -> MCPServer:
    return build_server(CheckerDeps(scam=scam_checker))


PipelineDepsFactory = Callable[..., PipelineDeps]


@pytest.fixture
def pipeline_deps(
    adapter: FakeAdapter,
    locales: Locales,
    app_config: AppConfig,
    fake_llm: FakeLLMClient,
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
    ) -> PipelineDeps:
        return PipelineDeps(
            adapters={"telegram": adapter},
            locales=locales,
            config=config or app_config,
            llm=llm or fake_llm,
            executor=Executor(caller or McpToolCaller(mcp_server, 5)),
            store=request_store or store,
            expander=NoopExpander(),
        )

    return build
