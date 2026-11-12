"""Per-request trace ID kept in a context variable and added to every log line."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

from structlog.typing import EventDict, WrappedLogger

TRACE_ID_KEY = "trace_id"

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    """Return a fresh random trace ID (32 lowercase hex characters)."""
    return uuid.uuid4().hex


def get_trace_id() -> str | None:
    """Return the trace ID of the current context, or ``None`` outside a trace."""
    return _trace_id.get()


def set_trace_id(trace_id: str) -> Token[str | None]:
    """Set the current trace ID and return a token for :func:`reset_trace_id`."""
    if not trace_id:
        msg = "trace_id must be a non-empty string"
        raise ValueError(msg)
    return _trace_id.set(trace_id)


def reset_trace_id(token: Token[str | None]) -> None:
    """Restore the trace ID that was current before the matching :func:`set_trace_id`."""
    _trace_id.reset(token)


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Run a block under ``trace_id``, restoring the previous ID after.

    A missing or empty ``trace_id`` (for example an absent header or job field) gets a fresh
    one, so tracing never fails the request it is attached to.
    """
    current = trace_id or new_trace_id()
    token = set_trace_id(current)
    try:
        yield current
    finally:
        reset_trace_id(token)


def add_trace_id(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Structlog processor: add the current trace ID unless the event already has one."""
    trace_id = _trace_id.get()
    if trace_id is not None:
        event_dict.setdefault(TRACE_ID_KEY, trace_id)
    return event_dict
