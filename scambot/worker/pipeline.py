"""End-to-end processing of one job: preprocess, analyse, plan, check, decide, reply, store."""

import asyncio
import datetime as dt
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

from scambot.agent.analyze import Analysis, analyze
from scambot.agent.executor import CheckInputs, Executor, input_hash, results_of
from scambot.agent.planner import Plan, build_plan
from scambot.agent.reply import Reply, ReplyInputs, choose_reply_language, compose_reply
from scambot.agent.verdict import decide
from scambot.channels.base import ChannelAdapter
from scambot.core.config import AppConfig
from scambot.core.contracts import CheckerResult, ExecutedCheck, Job, Preprocessed, Verdict
from scambot.core.llm import LLMClient, LLMUsage, cost_usd, usage_scope
from scambot.core.locales import Locales
from scambot.core.logging import get_logger
from scambot.core.store import NewRequest, RequestStatus, RequestStore, StartedRequest, StoredResult
from scambot.core.tracing import trace_context
from scambot.preprocess.entities import EntityLimits, UrlExpander, extract_entities
from scambot.preprocess.hashing import text_sha256

DEPS_KEY: Final = "pipeline"

_log = get_logger(component="pipeline")


@dataclass(frozen=True)
class PipelineDeps:
    adapters: Mapping[str, ChannelAdapter]
    locales: Locales
    config: AppConfig
    llm: LLMClient
    executor: Executor
    store: RequestStore
    expander: UrlExpander


@dataclass
class JobProgress:
    """What has been produced so far, kept for a partial reply when the deadline passes."""

    plan: Plan | None = None
    analysis: Analysis | None = None
    results: list[ExecutedCheck] = field(default_factory=list)


@dataclass(frozen=True)
class _Outcome:
    status: RequestStatus
    checks: list[ExecutedCheck]
    verdict: Verdict | None
    reply: Reply
    language: str


async def _guarded[T](stage: str, call: Awaitable[T]) -> T | None:
    """Run a store call; a failure is logged and never blocks the reply."""
    try:
        return await call
    except Exception as exc:  # the reply matters more than the record
        _log.warning("store_failed", stage=stage, error=type(exc).__name__)
        return None


async def _run_checks(deps: PipelineDeps, job: Job, progress: JobProgress) -> _Outcome:
    thresholds = deps.config.thresholds
    text = (job.text or "")[: thresholds.pipeline.max_text_chars]
    entities = await extract_entities(
        text,
        expander=deps.expander,
        shorteners=frozenset(deps.config.scam_patterns.shorteners),
        limits=EntityLimits(thresholds.entities.max_per_kind, thresholds.entities.expand_timeout_s),
    )
    pre = Preprocessed(kind=job.kind, text=text, entities=entities)
    progress.analysis = await analyze(
        pre,
        llm=deps.llm,
        received_on=job.received_at.date(),
        timeout_s=thresholds.pipeline.analyze_timeout_s,
    )
    progress.plan = build_plan(job.kind, pre, progress.analysis, deps.config.routing)
    checks = await deps.executor.run(
        progress.plan,
        CheckInputs(pre, progress.analysis, job.trace_id),
        on_result=progress.results.append,
    )
    verdict = decide(results_of(checks), job.kind, deps.config.verdict_rules)
    language = choose_reply_language(progress.analysis, job.preferred_lang, deps.locales)
    reply = await compose_reply(
        ReplyInputs(verdict, results_of(checks), language, use_llm=True),
        llm=deps.llm,
        locales=deps.locales,
        config=thresholds.reply,
        timeout_s=thresholds.pipeline.explanation_timeout_s,
    )
    return _Outcome("replied", checks, verdict, reply, language)


async def _partial(deps: PipelineDeps, job: Job, progress: JobProgress) -> _Outcome:
    """The reply when the deadline passed: a verdict from what finished, or an apology."""
    language = choose_reply_language(progress.analysis, job.preferred_lang, deps.locales)
    checks = list(progress.results)
    if not checks:
        reply = Reply(text=deps.locales.get(language).timeout, explanation_source="template")
        return _Outcome("timeout", checks, None, reply, language)
    finished = {check.result.checker for check in checks}
    for planned in progress.plan.checkers if progress.plan else ():
        if planned.name not in finished:
            missing = CheckerResult(
                checker=planned.name,
                status="error",
                signal="none",
                confidence=0.0,
                duration_ms=0,
                error="deadline",
            )
            checks.append(ExecutedCheck(result=missing, input_hash=input_hash({})))
    verdict = decide(results_of(checks), job.kind, deps.config.verdict_rules)
    reply = await compose_reply(
        ReplyInputs(verdict, results_of(checks), language, use_llm=False),
        llm=deps.llm,
        locales=deps.locales,
        config=deps.config.thresholds.reply,
        timeout_s=deps.config.thresholds.pipeline.explanation_timeout_s,
    )
    return _Outcome("timeout", checks, verdict, reply, language)


async def _checked(deps: PipelineDeps, job: Job) -> tuple[_Outcome, JobProgress]:
    progress = JobProgress()
    try:
        async with asyncio.timeout(deps.config.thresholds.pipeline.deadline_s) as deadline:
            return await _run_checks(deps, job, progress), progress
    except TimeoutError:
        if not deadline.expired():
            raise
    _log.warning("job_deadline", job_id=job.job_id, finished=len(progress.results))
    return await _partial(deps, job, progress), progress


def _cost(deps: PipelineDeps, usages: list[LLMUsage], checks: list[ExecutedCheck]) -> Decimal:
    remote: list[LLMUsage] = []
    for check in checks:
        for item in check.result.details.get("llm_usage", []) or []:
            try:
                remote.append(LLMUsage.model_validate(item))
            except ValueError:
                _log.warning("llm_usage_invalid", checker=check.result.checker)
    return cost_usd([*usages, *remote], deps.config.thresholds.llm_prices_usd_per_mtok)


async def _start(deps: PipelineDeps, job: Job) -> StartedRequest | None:
    new = NewRequest(
        job_id=job.job_id,
        user_hash=job.user_hash,
        channel=job.channel,
        kind=job.kind,
        sha256=text_sha256(job.text or ""),
        language=job.preferred_lang,
    )
    return await _guarded("start", deps.store.start(new))


def _log_reply(job: Job, outcome: _Outcome, started: float, cost: Decimal) -> None:
    verdict = outcome.verdict
    _log.info(
        "job_replied",
        job_id=job.job_id,
        status=outcome.status,
        label=verdict.label if verdict else None,
        band=verdict.band if verdict else None,
        score=verdict.score if verdict else None,
        deciding=verdict.deciding_checker if verdict else None,
        unavailable=len(verdict.unavailable) if verdict else 0,
        explanation_source=outcome.reply.explanation_source,
        duration_ms=round((time.perf_counter() - started) * 1000),
        cost_usd=str(cost),
    )


async def _finish(
    deps: PipelineDeps,
    job: Job,
    request: StartedRequest | None,
    done: tuple[_Outcome, JobProgress],
    usages: list[LLMUsage],
) -> tuple[_Outcome, Decimal]:
    """Store the result, deliver the reply, then mark the request with its final status."""
    outcome, progress = done
    cost = _cost(deps, usages, outcome.checks)
    if request is not None:
        analysis = progress.analysis
        stored = StoredResult(
            checks=outcome.checks,
            verdict=outcome.verdict,
            reply_text=outcome.reply.text,
            reply_language=outcome.language,
            claims=list(analysis.claims) if analysis else [],
            language=(analysis.reply_language or None) if analysis else None,
            cost_usd=cost,
        )
        await _guarded("save", deps.store.save(request.id, stored))
    await deps.adapters[job.channel].send_verdict(job, outcome.reply.text)
    if request is not None:
        now = dt.datetime.now(dt.UTC)
        await _guarded("set_status", deps.store.set_status(request.id, outcome.status, at=now))
    return outcome, cost


async def process_job(ctx: dict[str, Any], job: Job) -> None:
    """Queue entry point: produce the verdict for ``job`` and deliver it to the user."""
    deps: PipelineDeps = ctx[DEPS_KEY]
    started = time.perf_counter()
    with trace_context(job.trace_id), usage_scope() as usages:
        _log.info("job_started", job_id=job.job_id, kind=job.kind, attempt=ctx.get("job_try"))
        request = await _start(deps, job)
        if request is not None and request.status == "replied":
            _log.info("job_already_replied", job_id=job.job_id)
            return
        try:
            done = await _checked(deps, job)
            outcome, cost = await _finish(deps, job, request, done, usages)
        except Exception:  # includes a failed delivery, so the request is never left "received"
            if request is not None:
                await _guarded("set_status", deps.store.set_status(request.id, "failed"))
            raise
        _log_reply(job, outcome, started, cost)
