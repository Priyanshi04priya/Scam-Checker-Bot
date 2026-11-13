"""Fingerprinting a job's content and answering from an earlier finished verdict."""

import asyncio
import datetime as dt
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from scambot.agent.reply import Reply, ReplyInputs, compose_reply
from scambot.agent.verdict import decide
from scambot.core.contracts import Job
from scambot.core.locales import IST
from scambot.core.logging import get_logger
from scambot.core.media import MAX_MEDIA_BYTES, MediaRejected, MediaUnavailable
from scambot.core.media_store import media_key
from scambot.core.store import CachedVerdict, Fingerprint
from scambot.preprocess.base import run_blocking
from scambot.preprocess.hashing import file_sha256, image_phash, phash_to_int, text_sha256
from scambot.preprocess.media import HASH_TIMEOUT_S

if TYPE_CHECKING:
    from scambot.worker.pipeline import PipelineDeps

_log = get_logger(component="reuse")


async def download(deps: "PipelineDeps", job: Job, key: str) -> Path:
    """Fetch the job's file, retrying a transient failure once."""
    adapter = deps.adapters[job.channel]
    limits = deps.config.thresholds.media
    try:
        async with asyncio.timeout(limits.download_timeout_s) as budget:
            for attempt in (1, 2):
                try:
                    return await adapter.download_media(
                        job,
                        deps.media_store,
                        key,
                        max_bytes=MAX_MEDIA_BYTES,
                        timeout_s=limits.download_timeout_s,
                    )
                except MediaUnavailable as exc:
                    _log.warning("media_download_failed", attempt=attempt, error=str(exc))
                    if attempt == 2:
                        raise MediaRejected("unreadable") from None
    except TimeoutError:
        # An inner timeout must not escape as a bare TimeoutError: the deadline handler
        # would take it for a job-wide timeout.
        if budget.expired():
            raise MediaRejected("unreadable") from None
        raise
    raise AssertionError  # pragma: no cover - the loop always returns or raises


async def fingerprint(deps: "PipelineDeps", job: Job) -> Fingerprint:
    """Download any media and hash it. Raises ``MediaRejected`` when the file cannot be read."""
    if job.kind == "text":
        return Fingerprint("text", text_sha256(job.text or ""), None, None)
    if not job.media_ref:
        raise MediaRejected("unreadable")
    key = media_key(job.job_id)
    deps.media_store.delete(key)  # a retry starts from a clean slate
    path = await download(deps, job, key)
    pool = deps.media.pool
    try:
        digest = await run_blocking(pool, lambda: file_sha256(path), HASH_TIMEOUT_S)
    except TimeoutError:
        raise MediaRejected("unreadable") from None
    phash: int | None = None
    if job.kind == "image":
        try:
            phash = phash_to_int(
                await run_blocking(pool, lambda: image_phash(path), HASH_TIMEOUT_S)
            )
        except Exception as exc:  # an exact match can still work without the perceptual hash
            _log.warning("phash_failed", error=type(exc).__name__)
    return Fingerprint(job.kind, digest, phash, text_sha256(job.text or ""))


async def lookup(
    deps: "PipelineDeps", fp: Fingerprint, request_id: uuid.UUID | None
) -> CachedVerdict | None:
    """The earlier verdict for the same content, or None; a failure is only a miss."""
    if request_id is None or fp.sha256 is None:
        return None
    limits = deps.config.thresholds.cache
    try:
        if deps.runtime is not None:
            window = await deps.runtime.verdict_window()
        else:
            window = dt.timedelta(days=limits.verdict_window_days)
        return await deps.store.find_cached(
            fp,
            window=window,
            max_distance=limits.phash_max_distance,
            candidates=limits.phash_candidates,
            now=dt.datetime.now(dt.UTC),
            exclude=request_id,
        )
    except Exception as exc:  # the check can still be run in full
        _log.warning("cache_lookup_failed", error=type(exc).__name__)
        return None


async def cached_reply(deps: "PipelineDeps", job: Job, cached: CachedVerdict) -> tuple[Reply, str]:
    """The reply for a cache hit, in the user's language, led by a note saying when it was made."""
    language = deps.locales.resolve(job.user_lang or job.preferred_lang or cached.reply_language)
    text = cached.reply_text
    if language != cached.reply_language:
        results = await deps.store.load_results(cached.request_id)
        verdict = decide(results, job.kind, deps.config.verdict_rules)
        rebuilt = await compose_reply(
            ReplyInputs(verdict, results, language, use_llm=False),
            llm=deps.llm,
            locales=deps.locales,
            config=deps.config.thresholds.reply,
            timeout_s=deps.config.thresholds.pipeline.explanation_timeout_s,
        )
        text = rebuilt.text
    when = cached.checked_at.astimezone(IST).strftime("%d-%m-%Y %H:%M")
    note = deps.locales.get(language).cached.format(when=when)
    return Reply(text=f"{note}\n\n{text}", explanation_source="template"), language
