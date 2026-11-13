"""Runs the planned checkers as tool calls on the checker server, in dependency waves."""

import asyncio
import datetime as dt
import hashlib
import json
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Final, Protocol

from mcp import Client
from mcp.server.mcpserver import MCPServer

from scambot.agent.analyze import Analysis
from scambot.agent.planner import Plan, PlannedChecker
from scambot.core.contracts import CheckerName, CheckerResult, ExecutedCheck, Preprocessed
from scambot.core.logging import get_logger
from scambot.core.media_store import AUDIO_FILE, ORIGINAL

_log = get_logger(component="executor")

MAX_CAPTION_CHARS: Final = 1000


class ToolCallError(Exception):
    """A tool call returned no usable result; the message is a short fixed code."""


@dataclass(frozen=True)
class CheckInputs:
    pre: Preprocessed
    analysis: Analysis | None
    trace_id: str
    prior: Mapping[CheckerName, CheckerResult] = field(default_factory=dict)
    media_key: str | None = None
    received_on: dt.date | None = None
    max_search_frames: int = 3


ArgumentBuilder = Callable[[CheckInputs], dict[str, Any]]


def _scam_arguments(inputs: CheckInputs) -> dict[str, Any]:
    analysed = inputs.analysis.reply_language if inputs.analysis else None
    language = analysed or inputs.pre.language
    return {
        "text": inputs.pre.text,
        "entities": inputs.pre.entities.model_dump(mode="json"),
        "language": language,
        "trace_id": inputs.trace_id,
    }


def _factcheck_arguments(inputs: CheckInputs) -> dict[str, Any]:
    claims = list(inputs.analysis.claims) if inputs.analysis else []
    return {"claims": claims, "trace_id": inputs.trace_id}


def _iso(value: dt.date | None) -> str | None:
    return value.isoformat() if value else None


def _require_media_key(inputs: CheckInputs) -> str:
    if inputs.media_key is None:
        msg = "no_media_key"
        raise ValueError(msg)
    return inputs.media_key


def frame_names(pre: Preprocessed) -> list[str]:
    """File names of the pictures to inspect: the image itself, or a video's key frames."""
    if pre.kind == "video":
        return list(pre.media.keyframe_names) if pre.media else []
    return [ORIGINAL]


def pick_frames(names: Sequence[str], limit: int) -> list[str]:
    """At most ``limit`` names, evenly spaced, always including the first and the last."""
    if limit <= 0:
        return []
    if len(names) <= limit:
        return list(names)
    if limit == 1:
        return [names[0]]
    step = (len(names) - 1) / (limit - 1)
    return [names[round(i * step)] for i in range(limit)]


def _provenance_arguments(inputs: CheckInputs) -> dict[str, Any]:
    claimed = inputs.analysis.claimed_date if inputs.analysis else None
    return {
        "media_key": _require_media_key(inputs),
        "reference_date": _iso(claimed or inputs.received_on),
        "sha256": inputs.pre.media.sha256 if inputs.pre.media else None,
        "frames": pick_frames(frame_names(inputs.pre), inputs.max_search_frames),
        "trace_id": inputs.trace_id,
    }


def _caption_arguments(inputs: CheckInputs) -> dict[str, Any]:
    analysis = inputs.analysis
    provenance = inputs.prior.get("check_provenance")
    origin = None
    if provenance is not None and provenance.status == "ok":
        found = provenance.details.get("origin")
        origin = found if isinstance(found, dict) else None
    return {
        "caption": inputs.pre.text[:MAX_CAPTION_CHARS],
        "claimed_date": _iso(analysis.claimed_date if analysis else None),
        "claimed_place": analysis.claimed_place if analysis else None,
        "received_on": _iso(inputs.received_on),
        "origin": origin,
        "trace_id": inputs.trace_id,
    }


def _deepfake_arguments(inputs: CheckInputs) -> dict[str, Any]:
    return {
        "media_key": _require_media_key(inputs),
        "frames": frame_names(inputs.pre),
        "sha256": inputs.pre.media.sha256 if inputs.pre.media else None,
        "trace_id": inputs.trace_id,
    }


def _voice_arguments(inputs: CheckInputs) -> dict[str, Any]:
    return {
        "media_key": _require_media_key(inputs),
        "audio": AUDIO_FILE,
        "sha256": inputs.pre.media.sha256 if inputs.pre.media else None,
        "trace_id": inputs.trace_id,
    }


DEFAULT_BUILDERS: Final[Mapping[CheckerName, ArgumentBuilder]] = {
    "analyze_scam_patterns": _scam_arguments,
    "fact_check_claim": _factcheck_arguments,
    "check_provenance": _provenance_arguments,
    "check_caption_context": _caption_arguments,
    "detect_deepfake": _deepfake_arguments,
    "detect_voice_clone": _voice_arguments,
}


class ToolSession(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class ToolCaller(Protocol):
    def session(self) -> AbstractAsyncContextManager[ToolSession]: ...


class _McpSession:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._client.call_tool(name, arguments)
        if result.is_error:
            msg = "tool_error"
            raise ToolCallError(msg)
        if result.structured_content is None:
            msg = "no_structured_content"
            raise ToolCallError(msg)
        return dict(result.structured_content)


class McpToolCaller:
    """Calls tools over Streamable HTTP (a URL) or in-process (an ``MCPServer``)."""

    def __init__(self, target: str | MCPServer, connect_timeout_s: float) -> None:
        self._target = target
        self._connect_timeout_s = connect_timeout_s

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ToolSession]:
        client = Client(self._target)
        # asyncio.timeout, not wait_for: the client's task group must stay in this task.
        async with asyncio.timeout(self._connect_timeout_s):
            await client.__aenter__()
        try:
            yield _McpSession(client)
        finally:
            await client.__aexit__(None, None, None)


def input_hash(arguments: Mapping[str, Any]) -> str:
    """Stable hash of a tool call's arguments, ignoring the trace ID and the media key."""
    payload = {k: v for k, v in arguments.items() if k not in ("trace_id", "media_key")}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _error(name: str, code: str, started: float) -> CheckerResult:
    return CheckerResult(
        checker=name,
        status="error",
        signal="none",
        confidence=0.0,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error=code,
    )


def _waves(plan: Plan) -> list[list[PlannedChecker]]:
    """Topological layers of the plan, keeping plan order inside each layer."""
    done: set[CheckerName] = set()
    remaining = list(plan.checkers)
    waves: list[list[PlannedChecker]] = []
    while remaining:
        wave = [c for c in remaining if all(dep in done for dep in c.depends_on)]
        if not wave:  # pragma: no cover - the planner and config validation prevent cycles
            wave = remaining
        waves.append(wave)
        done.update(c.name for c in wave)
        remaining = [c for c in remaining if c not in wave]
    return waves


OnResult = Callable[[ExecutedCheck], None]


class Executor:
    def __init__(
        self,
        caller: ToolCaller,
        builders: Mapping[CheckerName, ArgumentBuilder] = DEFAULT_BUILDERS,
        max_search_frames: int = 3,
    ) -> None:
        self._caller = caller
        self._builders = builders
        self._max_search_frames = max_search_frames

    async def _call(
        self, session: ToolSession, checker: PlannedChecker, arguments: dict[str, Any]
    ) -> CheckerResult:
        raw = await asyncio.wait_for(session.call(checker.name, arguments), checker.timeout_s)
        result = CheckerResult.model_validate(raw)
        if result.checker != checker.name:
            msg = "wrong_checker"
            raise ToolCallError(msg)
        return result

    async def _one(
        self, session: ToolSession, checker: PlannedChecker, inputs: CheckInputs
    ) -> ExecutedCheck:
        started = time.perf_counter()
        builder = self._builders.get(checker.name)
        if builder is None:
            return ExecutedCheck(
                result=_error(checker.name, "not_implemented", started), input_hash=input_hash({})
            )
        arguments: dict[str, Any] = {}
        try:
            arguments = builder(inputs)
            result = await self._call(session, checker, arguments)
        except TimeoutError:
            result = _error(checker.name, "timeout", started)
        except ToolCallError as exc:
            result = _error(checker.name, str(exc), started)
        except Exception as exc:  # any other failure becomes an error result
            result = _error(checker.name, type(exc).__name__, started)
        return ExecutedCheck(result=result, input_hash=input_hash(arguments))

    async def _run_waves(
        self, session: ToolSession, plan: Plan, base: CheckInputs, report: OnResult
    ) -> None:
        base = replace(base, max_search_frames=self._max_search_frames)
        prior: dict[CheckerName, CheckerResult] = dict(base.prior)
        for wave in _waves(plan):
            inputs = replace(base, prior=dict(prior))

            async def run(checker: PlannedChecker, inputs: CheckInputs = inputs) -> None:
                executed = await self._one(session, checker, inputs)
                prior[checker.name] = executed.result
                report(executed)

            await asyncio.gather(*(run(checker) for checker in wave))

    async def run(
        self, plan: Plan, inputs: CheckInputs, on_result: OnResult | None = None
    ) -> list[ExecutedCheck]:
        """Run every planned checker; failures become ``error`` results, never exceptions."""
        executed: list[ExecutedCheck] = []

        def report(check: ExecutedCheck) -> None:
            executed.append(check)
            result = check.result
            _log.info(
                "checker_run",
                checker=result.checker,
                status=result.status,
                signal=result.signal,
                duration_ms=result.duration_ms,
                error=result.error,
            )
            if on_result is not None:
                on_result(check)

        if not plan.checkers:
            return executed
        started = time.perf_counter()
        connected = False
        try:
            async with self._caller.session() as session:
                connected = True
                await self._run_waves(session, plan, inputs, report)
        except Exception as exc:  # an unreachable checker server must not fail the job
            _log.warning("mcp_unavailable", error=type(exc).__name__, connected=connected)
        done = {check.result.checker for check in executed}
        for checker in plan.checkers:
            if checker.name not in done:
                report(
                    ExecutedCheck(
                        result=_error(checker.name, "mcp_unavailable", started),
                        input_hash=input_hash({}),
                    )
                )
        order: dict[str, int] = {name: i for i, name in enumerate(plan.names)}
        return sorted(executed, key=lambda check: order.get(check.result.checker, len(order)))


def results_of(checks: Sequence[ExecutedCheck]) -> list[CheckerResult]:
    return [check.result for check in checks]
