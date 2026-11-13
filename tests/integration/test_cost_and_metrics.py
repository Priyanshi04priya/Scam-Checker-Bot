"""Cost, timing, metrics and buttons for finished checks."""

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from arq.connections import ArqRedis

from scambot.core.config import AppConfig
from scambot.core.locales import Locales
from scambot.core.metrics import Metrics, render
from scambot.mcp_server.clients.sightengine import SynthScores
from tests.conftest import PipelineDepsFactory
from tests.fakes import FakeAdapter, FakeImageAnalyzer, FakeLLMClient, InMemoryStore
from tests.integration.test_image_checks import analysis_of, photo
from tests.integration.test_scam_text import check


@pytest.fixture(autouse=True)
def _skip_redis_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_info(*_args: Any) -> None:
        return None

    monkeypatch.setattr("arq.worker.log_redis_info", no_info)


@pytest.fixture(autouse=True)
def _llm(fake_llm: FakeLLMClient, analyzer: FakeImageAnalyzer) -> None:
    fake_llm.responses["analyze"] = analysis_of([])
    analyzer.default = SynthScores(0.93, 0.1)


async def test_cost_adds_vendor_calls_to_the_llm_cost_and_a_repeat_costs_nothing(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
    app_config: AppConfig,
) -> None:
    deps = replace(pipeline_deps(), metrics=Metrics(redis))
    prices = app_config.thresholds.costs.vendor_usd_per_call

    await check(redis, adapter, locales, deps, photo(adapter, 1))
    await check(redis, adapter, locales, deps, photo(adapter, 2))

    first, second = store.requests.values()
    calls = {c.result.checker: c.result.details.get("vendor_calls") for c in first.checks}
    assert calls["check_provenance"] == {"google_vision": 1}
    assert calls["detect_deepfake"] == {"sightengine": 1}
    llm = Decimal(str(first.cost_usd)) - prices["google_vision"] - prices["sightengine"]
    assert llm > 0  # the language-model share is what is left
    assert first.duration_ms is not None

    assert second.cache_of == first.id
    assert Decimal(str(second.cost_usd)) == 0
    assert second.duration_ms is not None
    assert all("vendor_calls" not in c.result.details for c in second.checks)

    text = (await render(redis)).decode()
    assert 'requests_total{channel="telegram",kind="image",label="possibly_ai"} 2.0' in text
    assert 'cache_hits_total{layer="verdict"} 1.0' in text
    assert 'api_cost_usd_total{vendor="sightengine"} 0.001' in text
    assert 'api_cost_usd_total{vendor="google_vision"} 0.0035' in text
    anthropic = next(line for line in text.splitlines() if 'vendor="anthropic"' in line)
    assert float(anthropic.split()[-1]) > 0
    assert 'stage_seconds_count{stage="total"} 2.0' in text


async def test_the_verdict_carries_buttons_with_the_stored_request_id(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    await check(redis, adapter, locales, pipeline_deps(), photo(adapter))

    (row,) = store.requests.values()
    (buttons,) = adapter.keyboards.values()
    (line,) = buttons
    texts = locales.get("en").buttons
    assert [b.label for b in line] == [texts.details, texts.helpful, texts.wrong]
    assert [b.data for b in line] == [f"d:{row.id}", f"h:{row.id}", f"w:{row.id}"]


async def test_no_buttons_when_the_store_is_down(
    redis: ArqRedis,
    adapter: FakeAdapter,
    locales: Locales,
    pipeline_deps: PipelineDepsFactory,
    store: InMemoryStore,
) -> None:
    store.fail = True

    await check(redis, adapter, locales, pipeline_deps(), photo(adapter))

    assert adapter.keyboards == {}
    assert adapter.log[-1][0] == "edit"
