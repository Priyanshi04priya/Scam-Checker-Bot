"""With any one outside service switched off, the user still gets an honest reply."""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest
from arq.connections import ArqRedis

from scambot.agent.executor import McpToolCaller
from scambot.core.config import AppConfig
from scambot.core.locales import Locales
from scambot.core.media_store import ORIGINAL
from scambot.mcp_server.checkers.factcheck import FactCheckChecker, FactCheckSources
from scambot.mcp_server.checkers.scam import ScamChecker
from scambot.mcp_server.clients.safebrowsing import SafeBrowsingError
from scambot.mcp_server.clients.sightengine import SightengineError
from scambot.mcp_server.clients.vision import VisionError
from scambot.mcp_server.server import CheckerDeps, build_server
from scambot.preprocess.media import MediaEngines
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    FailingIndex,
    FakeAdapter,
    FakeImageAnalyzer,
    FakeLLMClient,
    FakeReputation,
    FakeStt,
    FakeVoiceDetector,
    FakeWebSearch,
    InMemoryStore,
    KeywordEmbedder,
    fake_media_update,
    fake_update,
)
from tests.integration.test_image_checks import analysis_of
from tests.integration.test_scam_text import check
from tests.media import OGG_BYTES, textured_jpeg
from tests.samples import ELECTRICITY_EN, VOICE_SCAM_TEXT


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


@dataclass
class World:
    """Everything a scenario may break, and what it sends."""

    adapter: FakeAdapter
    fake_llm: FakeLLMClient
    web_search: FakeWebSearch
    analyzer: FakeImageAnalyzer
    voice_detector: FakeVoiceDetector
    app_config: AppConfig
    embedder: KeywordEmbedder
    checker_deps: CheckerDeps
    engines: MediaEngines


def image_update(world: World, lang: str = "en") -> dict[str, Any]:
    world.adapter.files["img"] = textured_jpeg()
    return fake_media_update(1, "image", "img", caption="Delhi flood today", lang=lang)


def voice_update(world: World, lang: str = "en") -> dict[str, Any]:
    world.adapter.files["vo"] = OGG_BYTES
    return fake_media_update(1, "voice", "vo", source="voice", lang=lang)


def search_down(world: World) -> dict[str, Any]:
    world.web_search.pages = VisionError("status_500")
    return image_update(world)


def analyzer_down(world: World) -> dict[str, Any]:
    world.analyzer.by_name[ORIGINAL] = SightengineError("status_500")
    return image_update(world)


def voice_down(world: World) -> dict[str, Any]:
    world.voice_detector.score_value = RuntimeError("model crashed")
    world.engines = dataclasses.replace(world.engines, stt=FakeStt(VOICE_SCAM_TEXT, "en"))
    return voice_update(world)


def llm_down(world: World) -> dict[str, Any]:
    world.fake_llm.responses.clear()  # every call raises LLMError
    return fake_update(1, ELECTRICITY_EN)


def factcheck_down(world: World) -> dict[str, Any]:
    world.fake_llm.responses["analyze"] = analysis_of(["The government will give free laptops"])
    sources = FactCheckSources(api=None, index=FailingIndex(), embedder=world.embedder)
    checker = FactCheckChecker(
        sources, world.fake_llm, world.app_config.thresholds.factcheck, world.app_config.sources
    )
    world.checker_deps = dataclasses.replace(world.checker_deps, factcheck=checker)
    return fake_update(1, "Free laptops for all students, forward this to everyone")


def reputation_down(world: World) -> dict[str, Any]:
    world.fake_llm.responses["analyze"] = analysis_of([])
    reputation = FakeReputation(unsafe=SafeBrowsingError("status_503"))
    scam = ScamChecker(
        world.app_config.scam_patterns,
        world.app_config.thresholds.scam,
        world.fake_llm,
        reputation,
    )
    world.checker_deps = dataclasses.replace(world.checker_deps, scam=scam)
    return fake_update(1, f"{ELECTRICITY_EN} Pay now at https://pay-electricity-now.example/bill")


# scenario -> (builder, checks that must be named as unavailable in the reply)
SCENARIOS: Mapping[str, tuple[Callable[[World], dict[str, Any]], set[str]]] = {
    "web_search": (search_down, {"check_provenance"}),
    "image_analyzer": (analyzer_down, {"detect_deepfake"}),
    "voice_detector": (voice_down, {"detect_voice_clone"}),
    "llm": (llm_down, set()),
    "factcheck": (factcheck_down, {"fact_check_claim"}),
    "safe_browsing": (reputation_down, set()),
}


@pytest.fixture
def world(
    adapter: FakeAdapter,
    fake_llm: FakeLLMClient,
    web_search: FakeWebSearch,
    analyzer: FakeImageAnalyzer,
    voice_detector: FakeVoiceDetector,
    app_config: AppConfig,
    embedder: KeywordEmbedder,
    checker_deps: CheckerDeps,
    media_engines: MediaEngines,
) -> World:
    fake_llm.responses["analyze"] = analysis_of([])
    return World(
        adapter,
        fake_llm,
        web_search,
        analyzer,
        voice_detector,
        app_config,
        embedder,
        checker_deps,
        media_engines,
    )


async def run(
    world: World,
    redis: ArqRedis,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    update: dict[str, Any],
) -> str:
    caller = McpToolCaller(build_server(world.checker_deps), 5)
    deps = pipeline_deps(caller=caller, engines=world.engines)
    return await check(redis, world.adapter, locales, deps, update)


def named(reply: str, locales: Locales, lang: str) -> set[str]:
    """The checks the reply says could not run."""
    texts = locales.get(lang).verdict
    if texts.unavailable.split("{")[0] not in reply:
        return set()
    names = texts.checkers.model_dump()
    return {
        key
        for key, text in names.items()
        if text in reply.split(texts.unavailable.split("{")[0])[1]
    }


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_each_service_off_in_turn_still_gets_a_reply(
    scenario: str,
    world: World,
    redis: ArqRedis,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    build, expected = SCENARIOS[scenario]
    update = build(world)

    reply = await run(world, redis, locales, pipeline_deps, update)

    (row,) = store.requests.values()
    assert row.status == "replied"
    assert row.verdict is not None
    assert reply.endswith(locales.get("en").disclaimer)
    assert named(reply, locales, "en") == expected
    erred = {c.result.checker for c in row.checks if c.result.status == "error"}
    assert erred == expected


async def test_an_outage_is_named_in_hindi_too(
    world: World,
    redis: ArqRedis,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
) -> None:
    world.web_search.pages = VisionError("status_500")
    world.fake_llm.responses["analyze"] = analysis_of([], language="hi")

    reply = await run(world, redis, locales, pipeline_deps, image_update(world, lang="hi"))

    assert named(reply, locales, "hi") == {"check_provenance"}
