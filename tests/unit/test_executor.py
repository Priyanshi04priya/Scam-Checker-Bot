"""Tests for the executor, using a test tool server called in-process."""

import asyncio
import datetime as dt
import time
from dataclasses import replace
from typing import Any

import pytest
from mcp.server.mcpserver import MCPServer

from scambot.agent.analyze import Analysis
from scambot.agent.executor import (
    DEFAULT_BUILDERS,
    ArgumentBuilder,
    CheckInputs,
    Executor,
    McpToolCaller,
    frame_names,
    input_hash,
)
from scambot.agent.planner import Plan, PlannedChecker
from scambot.core.contracts import (
    CheckerName,
    CheckerResult,
    ExecutedCheck,
    MediaFeatures,
    Preprocessed,
)
from tests.fakes import FailingToolCaller

DELAY = 0.2


def ok(name: str, **details: Any) -> CheckerResult:
    return CheckerResult(
        checker=name, status="ok", signal="none", confidence=0.9, duration_ms=1, details=details
    )


def build_test_server() -> MCPServer:
    server = MCPServer("test-checkers")

    @server.tool(name="check_provenance")
    async def provenance(trace_id: str) -> CheckerResult:
        await asyncio.sleep(DELAY)
        return ok("check_provenance")

    @server.tool(name="detect_deepfake")
    async def deepfake(trace_id: str) -> CheckerResult:
        await asyncio.sleep(DELAY)
        return ok("detect_deepfake")

    @server.tool(name="check_caption_context")
    async def caption(trace_id: str, prior_signal: str) -> CheckerResult:
        return ok("check_caption_context", prior_signal=prior_signal)

    @server.tool(name="detect_voice_clone")
    async def voice(trace_id: str) -> CheckerResult:
        await asyncio.sleep(5)
        return ok("detect_voice_clone")  # pragma: no cover

    @server.tool(name="fact_check_claim")
    async def factcheck(trace_id: str) -> CheckerResult:
        return ok("detect_deepfake")  # the wrong checker name

    @server.tool(name="analyze_scam_patterns")
    async def scam(trace_id: str) -> CheckerResult:
        msg = "boom"
        raise RuntimeError(msg)

    return server


def plain(inputs: CheckInputs) -> dict[str, Any]:
    return {"trace_id": inputs.trace_id}


def with_prior(inputs: CheckInputs) -> dict[str, Any]:
    return {"trace_id": inputs.trace_id, "prior_signal": inputs.prior["check_provenance"].signal}


BUILDERS: dict[CheckerName, ArgumentBuilder] = {
    "check_provenance": plain,
    "detect_deepfake": plain,
    "check_caption_context": with_prior,
    "detect_voice_clone": plain,
    "fact_check_claim": plain,
    "analyze_scam_patterns": plain,
}
INPUTS = CheckInputs(pre=Preprocessed(kind="image", has_media=True), analysis=None, trace_id="t1")


def plan(*checkers: PlannedChecker) -> Plan:
    return Plan(kind="image", checkers=checkers)


@pytest.fixture
def executor() -> Executor:
    return Executor(McpToolCaller(build_test_server(), 5), BUILDERS)


def by_name(checks: list[ExecutedCheck]) -> dict[str, CheckerResult]:
    return {check.result.checker: check.result for check in checks}


async def test_waves_run_in_parallel_and_pass_prior_results(executor: Executor) -> None:
    seen: list[str] = []
    started = time.perf_counter()

    checks = await executor.run(
        plan(
            PlannedChecker("check_provenance", 2),
            PlannedChecker("check_caption_context", 2, ("check_provenance",)),
            PlannedChecker("detect_deepfake", 2),
        ),
        INPUTS,
        on_result=lambda check: seen.append(check.result.checker),
    )

    elapsed = time.perf_counter() - started
    assert elapsed < DELAY * 1.9  # the two slow checks overlapped
    assert [c.result.checker for c in checks] == [
        "check_provenance",
        "check_caption_context",
        "detect_deepfake",
    ]
    assert seen[-1] == "check_caption_context"
    results = by_name(checks)
    assert results["check_caption_context"].details == {"prior_signal": "none"}
    assert all(r.status == "ok" for r in results.values())


async def test_failures_become_error_results(executor: Executor) -> None:
    checks = await executor.run(
        plan(
            PlannedChecker("detect_voice_clone", 0.05),
            PlannedChecker("fact_check_claim", 2),
            PlannedChecker("analyze_scam_patterns", 2),
            PlannedChecker("detect_deepfake", 2),
        ),
        INPUTS,
    )

    results = by_name(checks)
    assert results["detect_voice_clone"].error == "timeout"
    assert results["fact_check_claim"].error == "wrong_checker"
    assert results["analyze_scam_patterns"].error == "tool_error"
    assert results["detect_deepfake"].status == "ok"
    for name in ("detect_voice_clone", "fact_check_claim", "analyze_scam_patterns"):
        assert (results[name].status, results[name].signal) == ("error", "none")
        assert results[name].confidence == 0.0


async def test_missing_builder_is_not_implemented() -> None:
    executor = Executor(McpToolCaller(build_test_server(), 5), {})

    (check,) = await executor.run(plan(PlannedChecker("check_provenance", 2)), INPUTS)

    assert check.result.error == "not_implemented"


async def test_builder_errors_are_contained() -> None:
    def broken(inputs: CheckInputs) -> dict[str, Any]:
        return {"x": inputs.prior["check_provenance"]}

    executor = Executor(McpToolCaller(build_test_server(), 5), {"detect_deepfake": broken})

    (check,) = await executor.run(plan(PlannedChecker("detect_deepfake", 2)), INPUTS)

    assert check.result.error == "KeyError"


async def test_unreachable_server_marks_every_checker() -> None:
    executor = Executor(FailingToolCaller(), BUILDERS)
    seen: list[ExecutedCheck] = []

    checks = await executor.run(
        plan(PlannedChecker("check_provenance", 2), PlannedChecker("detect_deepfake", 2)),
        INPUTS,
        on_result=seen.append,
    )

    assert [c.result.error for c in checks] == ["mcp_unavailable", "mcp_unavailable"]
    assert len(seen) == 2


async def test_unreachable_url_is_mcp_unavailable() -> None:
    executor = Executor(McpToolCaller("http://127.0.0.1:9/mcp", 2), BUILDERS)

    (check,) = await executor.run(plan(PlannedChecker("detect_deepfake", 2)), INPUTS)

    assert check.result.error == "mcp_unavailable"


async def test_empty_plan_does_not_connect() -> None:
    assert await Executor(FailingToolCaller(), BUILDERS).run(plan(), INPUTS) == []


def test_fact_check_builder_sends_the_analysis_claims() -> None:
    build = DEFAULT_BUILDERS["fact_check_claim"]
    analysis = Analysis(
        claims=["claim one", "claim two"],
        reply_language="hi",
        claimed_date=None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )
    pre = Preprocessed(kind="text", text="message")

    with_claims = build(CheckInputs(pre=pre, analysis=analysis, trace_id="t-1"))
    without = build(CheckInputs(pre=pre, analysis=None, trace_id="t-2"))

    assert with_claims == {"claims": ["claim one", "claim two"], "trace_id": "t-1"}
    assert without == {"claims": [], "trace_id": "t-2"}


def test_input_hash_ignores_the_trace_id() -> None:
    first = input_hash({"text": "बिजली", "trace_id": "a", "n": 1})
    second = input_hash({"n": 1, "trace_id": "b", "text": "बिजली"})

    assert first == second
    assert first != input_hash({"text": "other", "n": 1})
    assert len(first) == 64


def image_inputs(**kwargs: Any) -> CheckInputs:
    media = MediaFeatures(mime_type="image/jpeg", size_bytes=10, sha256="a" * 64)
    pre = Preprocessed(kind="image", text="Delhi flood today", has_media=True, media=media)
    analysis = Analysis(
        claims=["c"],
        reply_language="en",
        claimed_date=dt.date(2026, 11, 13),
        claimed_place="Delhi",
        extra_checkers=[],
        reason="",
    )
    return CheckInputs(
        pre, analysis, "t-1", media_key="k1", received_on=dt.date(2026, 11, 12), **kwargs
    )


def test_image_check_arguments() -> None:
    inputs = image_inputs()

    provenance = DEFAULT_BUILDERS["check_provenance"](inputs)
    deepfake = DEFAULT_BUILDERS["detect_deepfake"](inputs)

    assert provenance == {
        "media_key": "k1",
        "image_name": "original",
        "reference_date": "2026-11-13",
        "sha256": "a" * 64,
        "trace_id": "t-1",
    }
    assert deepfake["frames"] == ["original"]
    assert deepfake["media_key"] == "k1"


def test_reference_date_falls_back_to_the_received_date() -> None:
    inputs = image_inputs()
    no_date = replace(inputs, analysis=None)

    assert DEFAULT_BUILDERS["check_provenance"](no_date)["reference_date"] == "2026-11-12"


def test_caption_arguments_carry_the_provenance_origin() -> None:
    origin = {"url": "https://x.example/a", "date": "2019-01-01", "credible": True}
    prior = {"check_provenance": ok("check_provenance", origin=origin)}

    arguments = DEFAULT_BUILDERS["check_caption_context"](image_inputs(prior=prior))

    assert arguments == {
        "caption": "Delhi flood today",
        "claimed_date": "2026-11-13",
        "claimed_place": "Delhi",
        "received_on": "2026-11-12",
        "origin": origin,
        "trace_id": "t-1",
    }


@pytest.mark.parametrize("status", ["error", "inconclusive"])
def test_caption_origin_is_dropped_unless_provenance_is_ok(status: str) -> None:
    failed = ok("check_provenance", origin={"url": "u"}).model_copy(update={"status": status})

    arguments = DEFAULT_BUILDERS["check_caption_context"](
        image_inputs(prior={"check_provenance": failed})
    )
    nothing = DEFAULT_BUILDERS["check_caption_context"](image_inputs())

    assert arguments["origin"] is None
    assert nothing["origin"] is None


def test_missing_media_key_raises_in_the_builder() -> None:
    inputs = replace(image_inputs(), media_key=None)

    with pytest.raises(ValueError, match="no_media_key"):
        DEFAULT_BUILDERS["detect_deepfake"](inputs)


def test_video_frames_follow_the_keyframes() -> None:
    media = MediaFeatures(
        mime_type="video/mp4",
        size_bytes=1,
        sha256="b" * 64,
        keyframe_phashes=["0" * 16] * 2,  # one hash failed
        keyframe_names=["kf_00.jpg", "kf_01.jpg", "kf_02.jpg"],
    )
    video = Preprocessed(kind="video", has_media=True, media=media)

    assert frame_names(video) == ["kf_00.jpg", "kf_01.jpg", "kf_02.jpg"]
    assert frame_names(Preprocessed(kind="video")) == []


def test_input_hash_ignores_the_media_key() -> None:
    assert input_hash({"media_key": "a", "sha256": "x"}) == input_hash(
        {"media_key": "b", "sha256": "x"}
    )
    assert input_hash({"media_key": "a", "sha256": "x"}) != input_hash(
        {"media_key": "a", "sha256": "y"}
    )
