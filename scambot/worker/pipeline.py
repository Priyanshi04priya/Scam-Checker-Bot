"""End-to-end processing of one job: preprocess, analyse, plan, check, decide, reply, store."""

import asyncio
import datetime as dt
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

from arq import Retry

from scambot.agent.analyze import Analysis, analyze
from scambot.agent.executor import CheckInputs, Executor, input_hash, results_of
from scambot.agent.planner import Plan, build_plan
from scambot.agent.reply import Reply, ReplyInputs, choose_reply_language, compose_reply
from scambot.agent.verdict import decide
from scambot.channels.base import ChannelAdapter
from scambot.core.config import AppConfig
from scambot.core.contracts import (
    CheckerResult,
    ExecutedCheck,
    Job,
    Preprocessed,
    RejectReason,
    Verdict,
)
from scambot.core.llm import LLMClient, LLMUsage, cost_usd, usage_scope
from scambot.core.locales import Locales
from scambot.core.logging import get_logger
from scambot.core.media import MediaRejected
from scambot.core.media_store import ORIGINAL, MediaStore, media_key
from scambot.core.runtime import RuntimeConfig
from scambot.core.store import (
    CachedVerdict,
    Fingerprint,
    NewRequest,
    RequestStatus,
    RequestStore,
    StartedRequest,
    StoredResult,
)
from scambot.core.tracing import trace_context
from scambot.preprocess.entities import EntityLimits, UrlExpander, extract_entities
from scambot.preprocess.hashing import phash_to_int, text_sha256
from scambot.preprocess.media import MediaEngines, compose_text, detect_language, extract_media
from scambot.worker.reuse import cached_reply, fingerprint, lookup

DEPS_KEY: Final = "pipeline"
# Tries per job (the first run and the retries); the queue is configured with the same number.
MAX_TRIES: Final = 3
RETRY_DELAY_S: Final = 5

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
    media_store: MediaStore
    media: MediaEngines
    runtime: RuntimeConfig | None = None


@dataclass
class JobProgress:
    """What has been produced so far, kept for a partial reply when the deadline passes."""

    plan: Plan | None = None
    analysis: Analysis | None = None
    results: list[ExecutedCheck] = field(default_factory=list)
    pre: Preprocessed | None = None
    fingerprint: Fingerprint | None = None


@dataclass(frozen=True)
class _Outcome:
    status: RequestStatus
    checks: list[ExecutedCheck]
    verdict: Verdict | None
    reply: Reply
    language: str
    cached: CachedVerdict | None = None


async def _guarded[T](stage: str, call: Awaitable[T]) -> T | None:
    """Run a store call; a failure is logged and never blocks the reply."""
    try:
        return await call
    except Exception as exc:  # the reply matters more than the record
        _log.warning("store_failed", stage=stage, error=type(exc).__name__)
        return None


async def _prepare(deps: PipelineDeps, job: Job, progress: JobProgress) -> Preprocessed:
    """Download and read any media, then extract entities and the language from the text."""
    thresholds = deps.config.thresholds
    caption = (job.text or "")[: thresholds.pipeline.max_text_chars]
    features = None
    if job.kind != "text":
        path = deps.media_store.path(media_key(job.job_id), ORIGINAL)
        features = await extract_media(job.kind, path, deps.media)
    text = compose_text(caption, features)[: thresholds.pipeline.max_text_chars]
    entities = await extract_entities(
        text,
        expander=deps.expander,
        shorteners=frozenset(deps.config.scam_patterns.shorteners),
        limits=EntityLimits(thresholds.entities.max_per_kind, thresholds.entities.expand_timeout_s),
    )
    language = await detect_language(
        text, features.speech_language if features else None, deps.media
    )
    pre = Preprocessed(
        kind=job.kind,
        text=text,
        caption=caption,
        entities=entities,
        language=language,
        has_media=features is not None,
        media=features,
    )
    progress.pre = pre
    return pre


def _rejected(deps: PipelineDeps, job: Job, reason: RejectReason) -> _Outcome:
    _log.info("media_rejected", job_id=job.job_id, reason=reason, stage="worker")
    text = getattr(deps.locales.get(job.preferred_lang).rejected, reason)
    reply = Reply(text=text, explanation_source="template")
    return _Outcome("replied", [], None, reply, deps.locales.resolve(job.preferred_lang))


async def _run_checks(
    deps: PipelineDeps, job: Job, progress: JobProgress, request: StartedRequest | None
) -> _Outcome:
    thresholds = deps.config.thresholds
    try:
        progress.fingerprint = await fingerprint(deps, job)
        cached = await lookup(deps, progress.fingerprint, request.id if request else None)
        if cached is not None:
            return await _reuse(deps, job, cached)
        pre = await _prepare(deps, job, progress)
    except MediaRejected as exc:
        return _rejected(deps, job, exc.reason)
    progress.analysis = await analyze(
        pre,
        llm=deps.llm,
        received_on=job.received_at.date(),
        timeout_s=thresholds.pipeline.analyze_timeout_s,
    )
    progress.plan = build_plan(job.kind, pre, progress.analysis, deps.config.routing)
    checks = await deps.executor.run(
        progress.plan,
        CheckInputs(
            pre,
            progress.analysis,
            job.trace_id,
            media_key=media_key(job.job_id),
            received_on=job.received_at.date(),
        ),
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


async def _reuse(deps: PipelineDeps, job: Job, cached: CachedVerdict) -> _Outcome:
    """Answer from an earlier finished verdict: nothing is analysed, checked or billed."""
    reply, language = await cached_reply(deps, job, cached)
    _log.info(
        "cache_hit",
        job_id=job.job_id,
        match=cached.match,
        distance=cached.distance,
        kind=job.kind,
        age_s=round((dt.datetime.now(dt.UTC) - cached.checked_at).total_seconds()),
    )
    return _Outcome("replied", [], None, reply, language, cached=cached)


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


async def _checked(
    deps: PipelineDeps, job: Job, request: StartedRequest | None
) -> tuple[_Outcome, JobProgress]:
    progress = JobProgress()
    try:
        async with asyncio.timeout(deps.config.thresholds.pipeline.deadline_s) as deadline:
            return await _run_checks(deps, job, progress, request), progress
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
        sha256=text_sha256(job.text or "") if job.kind == "text" else None,
        language=job.preferred_lang,
        caption_hash=None if job.kind == "text" else text_sha256(job.text or ""),
    )
    return await _guarded("start", deps.store.start(new))


def _log_reply(job: Job, outcome: _Outcome, started: float, cost: Decimal) -> None:
    verdict = outcome.verdict
    shown = verdict or outcome.cached  # both carry label, band and score
    _log.info(
        "job_replied",
        job_id=job.job_id,
        status=outcome.status,
        cache=outcome.cached is not None,
        label=shown.label if shown else None,
        band=shown.band if shown else None,
        score=shown.score if shown else None,
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
    fp = progress.fingerprint
    if request is not None and outcome.cached is not None and fp is not None:
        await _guarded("save", deps.store.save_cached(request.id, outcome.cached, fp))
    elif request is not None:
        analysis = progress.analysis
        media = progress.pre.media if progress.pre else None
        phash = fp.phash if fp else None
        if phash is None and media and media.phash:
            phash = phash_to_int(media.phash)
        stored = StoredResult(
            checks=outcome.checks,
            verdict=outcome.verdict,
            reply_text=outcome.reply.text,
            reply_language=outcome.language,
            claims=list(analysis.claims) if analysis else [],
            language=(analysis.reply_language if analysis else None)
            or (progress.pre.language if progress.pre else None),
            cost_usd=cost,
            sha256=(fp.sha256 if fp else None) or (media.sha256 if media else None),
            phash=phash,
        )
        await _guarded("save", deps.store.save(request.id, stored))
    await deps.adapters[job.channel].send_verdict(job, outcome.reply.text)
    if request is not None:
        now = dt.datetime.now(dt.UTC)
        await _guarded("set_status", deps.store.set_status(request.id, outcome.status, at=now))
    return outcome, cost


def _delete_media(deps: PipelineDeps, job: Job) -> None:
    """Remove the job's downloaded file. Synchronous, so it also runs when the job is cancelled."""
    if job.kind == "text":
        return
    try:
        deps.media_store.delete(media_key(job.job_id))
    except Exception as exc:
        _log.warning("media_delete_failed", error=type(exc).__name__)


async def _retry_or_give_up(deps: PipelineDeps, job: Job, attempt: int) -> None:
    """Ask the queue to run the job again, or on the last try apologise to the user."""
    if attempt < MAX_TRIES:
        _log.warning("job_retry", job_id=job.job_id, attempt=attempt)
        raise Retry(defer=RETRY_DELAY_S)
    _log.error("job_gave_up", job_id=job.job_id, attempt=attempt)
    try:
        await deps.adapters[job.channel].send_verdict(
            job, deps.locales.get(job.preferred_lang).timeout
        )
    except Exception as exc:  # best effort: the original error is the one worth reporting
        _log.warning("apology_failed", error=type(exc).__name__)


async def process_job(ctx: dict[str, Any], job: Job) -> None:
    """Queue entry point: produce the verdict for ``job`` and deliver it to the user."""
    deps: PipelineDeps = ctx[DEPS_KEY]
    started = time.perf_counter()
    with trace_context(job.trace_id), usage_scope() as usages:
        try:
            _log.info("job_started", job_id=job.job_id, kind=job.kind, attempt=ctx.get("job_try"))
            request = await _start(deps, job)
            if request is not None and request.status == "replied":
                _log.info("job_already_replied", job_id=job.job_id)
                return
            try:
                done = await _checked(deps, job, request)
                outcome, cost = await _finish(deps, job, request, done, usages)
            except Exception:  # includes a failed delivery, so the request is never left "received"
                if request is not None:
                    await _guarded("set_status", deps.store.set_status(request.id, "failed"))
                await _retry_or_give_up(deps, job, ctx.get("job_try", 1))
                raise
            _log_reply(job, outcome, started, cost)
        finally:
            _delete_media(deps, job)
