"""Structured JSON logging with redaction of secrets and user message content."""

import dataclasses
import logging
import re
import sys
from collections.abc import Mapping
from typing import Any, Final, TextIO, cast

import structlog
from pydantic import BaseModel, SecretBytes, SecretStr
from structlog.typing import EventDict, FilteringBoundLogger, Processor, WrappedLogger

from scambot.core.tracing import add_trace_id

REDACTED: Final = "[REDACTED]"

# A key is secret when its last "_"-separated part is one of these words (optionally followed
# by a numeric suffix, as in ``x_hub_signature_256``), it ends with one of the compound names,
# or it is exactly ``key`` (URL query parameter). Matching the last part only keeps counters
# such as ``input_tokens`` or ``token_count`` and names such as ``cache_key`` visible.
_SECRET_KEY_RE: Final = re.compile(
    r"^key$|(?:^|_)(?:token|secret|password|passwd|pepper|apikey|authorization|cookie|"
    r"credential|credentials|signature|api_key|password_hash|secret_key|private_key|"
    r"access_key|signing_key)(?:_\d+)?$"
)

# A key carries user content (message text, captions, transcripts, OCR output, extracted
# claims) when its last "_"-separated part is one of these words, e.g. ``ocr_text``.
TEXT_KEYS: Final = frozenset(
    {
        "text",
        "caption",
        "transcript",
        "body",
        "content",
        "explanation",
        "claim",
        "claims",
    }
)

_VALUE_PATTERNS: Final = (
    # user:password@ inside URLs (database and Redis DSNs); the password may contain "@".
    # The lookbehind and the bounded scheme keep this linear on long runs of letters.
    (
        re.compile(
            r"(?<![a-z0-9+.\-])(?P<scheme>[a-z][a-z0-9+.\-]{0,31}://)[^\s:/@]*:[^\s/]+@",
            re.IGNORECASE,
        ),
        r"\g<scheme>" + REDACTED + "@",
    ),
    # Credentials passed as URL query parameters, with or without a prefix: ?key=...,
    # &api_key=..., &access_token=..., &api_secret=..., &hub.verify_token=..., ?password=...
    (
        re.compile(
            r"(?P<param>[?&](?:[a-z0-9_.\-]*[_.\-])?"
            r"(?:key|apikey|token|secret|password|passwd|signature)=)[^&#\s'\"]+",
            re.IGNORECASE,
        ),
        r"\g<param>" + REDACTED,
    ),
    # Telegram bot tokens: <bot id>:<35 url-safe characters>, also inside /bot<token>/ URLs
    (re.compile(r"(?<!\d)\d{6,12}:[A-Za-z0-9_-]{30,}"), REDACTED),
    # Anthropic API keys
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{10,}"), REDACTED),
    # Google API keys
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), REDACTED),
    # Bearer tokens in header values
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer " + REDACTED),
)


def _normalise_key(key: object) -> str:
    return str(key).lower().replace("-", "_")


def is_sensitive_key(key: object) -> bool:
    """Return True when values stored under ``key`` must never be logged."""
    name = _normalise_key(key)
    return name.rpartition("_")[2] in TEXT_KEYS or _SECRET_KEY_RE.search(name) is not None


def scrub_string(value: str) -> str:
    """Mask secret-looking substrings (URL credentials, bot tokens, API keys) in ``value``."""
    for pattern, replacement in _VALUE_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _to_plain(value: object) -> object:
    """Convert an object JSON cannot encode the way the renderer would, but before redaction.

    Models and dataclasses become dicts so their fields get key-based redaction too; anything
    else becomes its ``__structlog__()`` or ``repr()``, which the caller then scrubs.
    """
    if isinstance(value, BaseModel):
        return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    structlog_repr = getattr(value, "__structlog__", None)
    return structlog_repr() if callable(structlog_repr) else repr(value)


def _redact_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, SecretStr | SecretBytes):
        return REDACTED
    if isinstance(value, str):
        return scrub_string(value)
    if isinstance(value, Mapping):
        return {k: REDACTED if is_sensitive_key(k) else _redact_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_redact_value(item) for item in value]
    return _redact_value(_to_plain(value))


def redact_processor(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Structlog processor: replace sensitive values before the event is rendered."""
    for key, value in event_dict.items():
        if key != "event" and is_sensitive_key(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact_value(value)
    return event_dict


# The only levels structlog's filtering loggers support.
_SUPPORTED_LEVELS: Final = frozenset(
    {logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
)


def _level_number(level: str | int) -> int:
    number = level if isinstance(level, int) else logging.getLevelNamesMapping().get(level.upper())
    if isinstance(level, bool) or number is None or number not in _SUPPORTED_LEVELS:
        msg = f"unknown log level: {level!r}"
        raise ValueError(msg)
    return number


def configure_logging(level: str | int = "INFO", stream: TextIO | None = None) -> None:
    """Configure structlog to write one JSON object per line to ``stream`` (default stdout)."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_trace_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(
            structlog.tracebacks.ExceptionDictTransformer(show_locals=False)
        ),
        redact_processor,
        structlog.processors.JSONRenderer(),
    ]
    # Their request lines carry URLs, and media URLs carry access tokens.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(_level_number(level)),
        logger_factory=structlog.WriteLoggerFactory(file=stream or sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(**initial_values: Any) -> FilteringBoundLogger:
    """Return a structlog logger, optionally pre-bound with ``initial_values``."""
    return cast(FilteringBoundLogger, structlog.get_logger(**initial_values))
