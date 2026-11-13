"""Image checks end to end: intake, queue, worker, in-process checker server, fake services."""

import datetime as dt
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from arq.connections import ArqRedis

from scambot.agent.analyze import Analysis
from scambot.agent.executor import McpToolCaller
from scambot.core.locales import Locales
from scambot.factcheck_index.search import InMemoryFactcheckIndex
from scambot.mcp_server.clients.sightengine import SynthScores
from scambot.mcp_server.clients.vision import VisionError, WebPage
from scambot.mcp_server.server import CheckerDeps, build_server
from tests.conftest import PipelineDepsFactory
from tests.fakes import (
    FakeAdapter,
    FakeImageAnalyzer,
    FakeLLMClient,
    FakePageFetcher,
    FakeWebSearch,
    InMemoryStore,
    KeywordEmbedder,
    fake_media_update,
    judge,
    mismatch_reply,
    seed,
)
from tests.integration.test_scam_text import check
from tests.media import exif_jpeg, is_empty
from tests.samples import KOCHI_PAGE, KOCHI_URL, UNRELATED_FACTCHECKS

FLOOD_CLAIM = "A video shows floods in the city this week"
FLOOD_DEBUNK = UNRELATED_FACTCHECKS[0]
TODAY = dt.datetime.now(dt.UTC).date()


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


def analysis_of(
    claims: list[str],
    *,
    language: str = "en",
    claimed_date: dt.date | None = None,
    claimed_place: str | None = None,
) -> Any:
    def answer(_variables: Mapping[str, str]) -> Analysis:
        return Analysis(
            claims=claims,
            reply_language=language,
            claimed_date=claimed_date,
            claimed_place=claimed_place,
            extra_checkers=[],
            reason="",
        )

    return answer


def photo(adapter: FakeAdapter, update_id: int = 1, caption: str | None = None) -> dict[str, Any]:
    adapter.files["img"] = exif_jpeg()
    return fake_media_update(update_id, "image", "img", caption=caption)


def checks_of(store: InMemoryStore) -> dict[str, Any]:
    (row,) = store.requests.values()
    return {c.result.checker: c.result for c in row.checks}


def has_no_genuine_claim(reply: str) -> bool:
    return not any(word in reply.lower() for word in ("genuine", "authentic", "is real"))


@pytest.fixture
def old_flood(
    web_search: FakeWebSearch,
    page_fetcher: FakePageFetcher,
    fake_llm: FakeLLMClient,
    factcheck_index: InMemoryFactcheckIndex,
) -> None:
    web_search.pages = [WebPage(KOCHI_URL, "Kochi floods", "full")]
    page_fetcher.pages[KOCHI_URL] = KOCHI_PAGE
    fake_llm.responses["caption_compare"] = mismatch_reply(("place", "Delhi", "Kochi"))
    fake_llm.responses["factcheck_relevance"] = judge(about="flood")


@pytest.mark.parametrize("lang", ["en", "hi"])
@pytest.mark.usefixtures("old_flood")
async def test_old_picture_with_a_debunk_links_both_sources(
    lang: str,
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    factcheck_index: InMemoryFactcheckIndex,
    embedder: KeywordEmbedder,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    await seed(factcheck_index, embedder, FLOOD_DEBUNK)
    fake_llm.responses["analyze"] = analysis_of(
        [FLOOD_CLAIM], language=lang, claimed_date=TODAY, claimed_place="Delhi"
    )
    update = photo(adapter, caption="Delhi flood today")
    update["lang"] = lang

    reply = await check(redis, adapter, locales, pipeline_deps(), update)

    verdict = locales.get(lang).verdict
    heading = reply.splitlines()[0]
    assert verdict.bands.medium in heading
    assert verdict.labels.old_or_out_of_context in heading
    assert KOCHI_URL in reply
    assert FLOOD_DEBUNK.url in reply
    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.label == "old_or_out_of_context"
    assert row.verdict.deciding_checker in ("check_provenance", "check_caption_context")
    assert row.verdict.source_confirmed is False
    assert checks_of(store)["check_caption_context"].signal == "out_of_context"
    assert is_empty(media_root)


async def test_unknown_origin_and_a_synthetic_score_is_possibly_ai(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    analyzer: FakeImageAnalyzer,
    media_root: Path,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    analyzer.default = SynthScores(0.93, 0.1)

    reply = await check(redis, adapter, locales, pipeline_deps(), photo(adapter))

    assert "Medium risk · Possibly AI-generated" in reply.splitlines()[0]
    assert locales.get("en").verdict.explanations.possibly_ai in reply
    assert has_no_genuine_claim(reply)
    assert is_empty(media_root)


async def test_nothing_found_is_unknown(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    store: InMemoryStore,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])

    reply = await check(redis, adapter, locales, pipeline_deps(), photo(adapter))

    assert "Unknown · Could not verify" in reply.splitlines()[0]
    assert "No issues" not in reply
    assert "http" not in reply
    assert checks_of(store)["check_provenance"].status == "inconclusive"


async def test_search_failure_is_named_as_unavailable(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    web_search: FakeWebSearch,
    store: InMemoryStore,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    web_search.pages = VisionError("status_500")

    reply = await check(redis, adapter, locales, pipeline_deps(), photo(adapter))

    en = locales.get("en").verdict
    assert en.unavailable.replace("{checks}", en.checkers.check_provenance) in reply
    assert en.labels.could_not_verify in reply.splitlines()[0]
    results = checks_of(store)
    assert results["check_provenance"].status == "error"
    assert results["detect_deepfake"].status == "ok"


async def test_missing_analyzer_is_named_as_unavailable(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    checker_deps: CheckerDeps,
    store: InMemoryStore,
) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    no_analyzer = replace(checker_deps, deepfake=replace_analyzer(checker_deps))
    caller = McpToolCaller(build_server(no_analyzer), 5)

    reply = await check(redis, adapter, locales, pipeline_deps(caller=caller), photo(adapter))

    en = locales.get("en").verdict
    assert en.unavailable.replace("{checks}", en.checkers.detect_deepfake) in reply
    assert checks_of(store)["detect_deepfake"].error == "not_configured"


def replace_analyzer(deps: CheckerDeps) -> Any:
    from scambot.mcp_server.checkers.deepfake import DeepfakeChecker  # noqa: PLC0415

    old = deps.deepfake
    return DeepfakeChecker(None, old._media, old._limits)


async def test_no_keys_at_all_still_replies(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    checker_deps: CheckerDeps,
    store: InMemoryStore,
    media_root: Path,
) -> None:
    from scambot.mcp_server.checkers.provenance import ProvenanceChecker  # noqa: PLC0415

    fake_llm.responses["analyze"] = analysis_of([])
    old = checker_deps.provenance
    bare = ProvenanceChecker(None, old._fetcher, old._media, old._limits, old._sources)
    deps = replace(checker_deps, provenance=bare, deepfake=replace_analyzer(checker_deps))
    caller = McpToolCaller(build_server(deps), 5)

    reply = await check(redis, adapter, locales, pipeline_deps(caller=caller), photo(adapter))

    en = locales.get("en").verdict
    assert en.checkers.check_provenance in reply
    assert en.checkers.detect_deepfake in reply
    results = checks_of(store)
    assert results["check_provenance"].error == "not_configured"
    assert results["detect_deepfake"].error == "not_configured"
    assert is_empty(media_root)


async def test_credible_recent_page_with_a_consistent_caption_has_no_issues(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    fake_llm: FakeLLMClient,
    web_search: FakeWebSearch,
    page_fetcher: FakePageFetcher,
    store: InMemoryStore,
) -> None:
    recent = KOCHI_PAGE.replace("2019-08-10", TODAY.isoformat())
    web_search.pages = [WebPage(KOCHI_URL, "Kochi floods", "full")]
    page_fetcher.pages[KOCHI_URL] = recent
    fake_llm.responses["analyze"] = analysis_of([], claimed_place="Kochi")
    fake_llm.responses["caption_compare"] = mismatch_reply()

    reply = await check(
        redis, adapter, locales, pipeline_deps(), photo(adapter, caption="Kochi flood today")
    )

    assert "No issues found" in reply.splitlines()[0]
    assert has_no_genuine_claim(reply)
    (row,) = store.requests.values()
    assert row.verdict is not None
    assert row.verdict.source_confirmed is True
