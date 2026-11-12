"""Shared wrapper for checker tools: tracing, timing and failure isolation."""

import functools
import time
from collections.abc import Awaitable, Callable

from scambot.core.contracts import CheckerName, CheckerResult
from scambot.core.logging import get_logger
from scambot.core.tracing import trace_context

_log = get_logger(component="checker")

type CheckerFn[**P] = Callable[P, Awaitable[CheckerResult]]


def checker_tool[**P](name: CheckerName) -> Callable[[CheckerFn[P]], CheckerFn[P]]:
    """Run the tool inside the caller's trace, fill in ``duration_ms``, and turn any exception
    into an ``error`` result that names only the exception class (never its message)."""

    def decorate(func: CheckerFn[P]) -> CheckerFn[P]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> CheckerResult:
            trace_id = kwargs.get("trace_id")
            started = time.perf_counter()
            with trace_context(trace_id if isinstance(trace_id, str) else None):
                try:
                    result = await func(*args, **kwargs)
                    ms = round((time.perf_counter() - started) * 1000)
                    result = result.model_copy(update={"duration_ms": ms})
                except Exception as exc:  # a failing checker reports an error, never crashes
                    ms = round((time.perf_counter() - started) * 1000)
                    result = CheckerResult(
                        checker=name,
                        status="error",
                        signal="none",
                        confidence=0.0,
                        duration_ms=ms,
                        error=type(exc).__name__,
                    )
                _log.info(
                    "checker_done",
                    checker=name,
                    status=result.status,
                    signal=result.signal,
                    duration_ms=result.duration_ms,
                )
            return result

        return wrapper

    return decorate
