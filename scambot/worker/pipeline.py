"""End-to-end processing of one job."""

from typing import Any, Final

from scambot.channels.base import ChannelAdapter
from scambot.core.contracts import Job, Verdict
from scambot.core.locales import Locales
from scambot.core.logging import get_logger
from scambot.core.tracing import trace_context

ADAPTERS_KEY: Final = "adapters"
LOCALES_KEY: Final = "locales"

# Fixed result until the checkers exist: nothing has been checked, so nothing can be claimed.
STUB_VERDICT: Final = Verdict(label="could_not_verify", band="unknown", score=50)

_log = get_logger(component="pipeline")


def render_reply(verdict: Verdict, locales: Locales, language: str | None) -> str:
    """Reply text for ``verdict``: headline, note and disclaimer in the user's language."""
    texts = locales.get(language)
    band = getattr(texts.verdict.bands, verdict.band)
    label = getattr(texts.verdict.labels, verdict.label)
    return f"{band} · {label}\n\n{texts.verdict.stub_note}\n\n{texts.disclaimer}"


async def process_job(ctx: dict[str, Any], job: Job) -> None:
    """Queue entry point: produce the verdict for ``job`` and deliver it to the user."""
    adapters: dict[str, ChannelAdapter] = ctx[ADAPTERS_KEY]
    locales: Locales = ctx[LOCALES_KEY]
    with trace_context(job.trace_id):
        _log.info("job_started", job_id=job.job_id, kind=job.kind, attempt=ctx.get("job_try"))
        text = render_reply(STUB_VERDICT, locales, job.preferred_lang)
        await adapters[job.channel].send_verdict(job, text)
        _log.info("job_replied", job_id=job.job_id, label=STUB_VERDICT.label)
