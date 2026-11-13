"""The executor serves repeated checker calls from the cache."""

import dataclasses
import datetime as dt
from typing import Any

import pytest
from arq.connections import ArqRedis
from mcp.server.mcpserver import MCPServer

from scambot.agent.analyze import Analysis
from scambot.agent.executor import CheckInputs, Executor, McpToolCaller
from scambot.agent.planner import Plan, PlannedChecker
from scambot.core.cache import CheckerCache
from scambot.core.contracts import CheckerResult, ExecutedCheck, MediaFeatures, Preprocessed

TTLS = dict.fromkeys(
    (
        "analyze_scam_patterns",
        "fact_check_claim",
        "check_provenance",
        "check_caption_context",
        "detect_deepfake",
        "detect_voice_clone",
    ),
    1000,
)


class Vendor:
    """Counts calls and answers with a chosen status."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.status = "ok"

    def result(self, name: str) -> CheckerResult:
        self.calls.append(name)
        return CheckerResult(
            checker=name,
            status=self.status,  # type: ignore[arg-type]
            signal="none",
            confidence=0.5,
            duration_ms=40,
            error="vendor" if self.status == "error" else None,
            details={"llm_usage": [{"model": "m", "input_tokens": 9, "output_tokens": 9}]},
        )


@pytest.fixture
def vendor() -> Vendor:
    return Vendor()


@pytest.fixture
def server(vendor: Vendor) -> MCPServer:
    server = MCPServer("counting")

    @server.tool(name="check_provenance")
    async def provenance(
        media_key: str, reference_date: str, sha256: str, frames: list[str], trace_id: str
    ) -> CheckerResult:
        return vendor.result("check_provenance")

    @server.tool(name="detect_deepfake")
    async def deepfake(
        media_key: str, frames: list[str], sha256: str, trace_id: str
    ) -> CheckerResult:
        return vendor.result("detect_deepfake")

    @server.tool(name="check_caption_context")
    async def caption(
        caption: str,
        claimed_date: str | None,
        claimed_place: str | None,
        received_on: str,
        origin: dict[str, Any] | None,
        trace_id: str,
    ) -> CheckerResult:
        return vendor.result("check_caption_context")

    return server


@pytest.fixture
def executor(server: MCPServer, redis: ArqRedis) -> Executor:
    return Executor(McpToolCaller(server, 5), cache=CheckerCache(redis, TTLS, 60))  # type: ignore[arg-type]


PLAN = Plan(
    kind="image",
    checkers=(
        PlannedChecker("check_provenance", 2),
        PlannedChecker("detect_deepfake", 2),
        PlannedChecker("check_caption_context", 2, ("check_provenance",)),
    ),
)


def inputs(caption: str = "flood", day: int = 13, claimed: bool = False) -> CheckInputs:
    media = MediaFeatures(mime_type="image/jpeg", size_bytes=10, sha256="a" * 64)
    pre = Preprocessed(kind="image", text=caption, has_media=True, media=media)
    analysis = Analysis(
        claims=[],
        reply_language="en",
        claimed_date=dt.date(2020, 1, 1) if claimed else None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )
    return CheckInputs(pre, analysis, "t", media_key="k", received_on=dt.date(2026, 11, day))


def by_name(checks: list[ExecutedCheck]) -> dict[str, ExecutedCheck]:
    return {check.result.checker: check for check in checks}


async def test_a_repeat_is_served_from_the_cache_without_calling_the_vendor(
    executor: Executor, vendor: Vendor
) -> None:
    first = await executor.run(PLAN, inputs())
    second = await executor.run(PLAN, inputs())

    assert len(vendor.calls) == 3
    assert not any(check.cached for check in first)
    assert all(check.cached for check in second)
    for check in second:
        assert check.result.details["cached"] is True
        assert "llm_usage" not in check.result.details  # nothing to bill again
        assert check.input_hash


async def test_the_same_picture_on_another_day_still_hits(
    executor: Executor, vendor: Vendor
) -> None:
    await executor.run(PLAN, inputs(day=13))
    await executor.run(PLAN, inputs(day=14))

    # The caption check depends on the day it was received; the picture checks do not.
    assert vendor.calls.count("check_provenance") == 1
    assert vendor.calls.count("detect_deepfake") == 1


async def test_a_claimed_date_keeps_the_day_in_the_key(executor: Executor, vendor: Vendor) -> None:
    await executor.run(PLAN, inputs(claimed=True))
    await executor.run(PLAN, inputs(claimed=True))

    assert len(vendor.calls) == 3


async def test_a_new_caption_reuses_picture_checks_but_not_the_caption_check(
    executor: Executor, vendor: Vendor
) -> None:
    await executor.run(PLAN, inputs("flood in Delhi"))
    vendor.calls.clear()

    checks = by_name(await executor.run(PLAN, inputs("fire in Mumbai")))

    assert vendor.calls == ["check_caption_context"]
    assert checks["check_provenance"].cached
    assert checks["detect_deepfake"].cached
    assert not checks["check_caption_context"].cached


async def test_an_error_is_not_cached(executor: Executor, vendor: Vendor) -> None:
    vendor.status = "error"
    first = await executor.run(PLAN, inputs())
    vendor.status = "ok"
    second = await executor.run(PLAN, inputs())

    assert all(check.result.status == "error" for check in first)
    assert all(check.result.status == "ok" and not check.cached for check in second)
    assert len(vendor.calls) == 6


async def test_inconclusive_is_cached_with_the_short_ttl(
    executor: Executor, vendor: Vendor, redis: ArqRedis
) -> None:
    vendor.status = "inconclusive"
    await executor.run(PLAN, inputs())
    second = await executor.run(PLAN, inputs())

    assert len(vendor.calls) == 3
    assert all(check.cached for check in second)
    keys = [key async for key in redis.scan_iter(match="chk:*")]
    assert {await redis.ttl(key) for key in keys} <= set(range(1, 61))


async def test_without_a_cache_every_call_reaches_the_vendor(
    server: MCPServer, vendor: Vendor
) -> None:
    plain = Executor(McpToolCaller(server, 5))

    await plain.run(PLAN, inputs())
    await plain.run(PLAN, inputs())

    assert len(vendor.calls) == 6
    assert dataclasses.is_dataclass(PLAN)
