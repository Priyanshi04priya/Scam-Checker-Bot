"""Tests for JSON logging and redaction."""

import dataclasses
import io
import json
import logging
import time
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from scambot.core.logging import (
    REDACTED,
    configure_logging,
    get_logger,
    is_sensitive_key,
    redact_processor,
    scrub_string,
)
from scambot.core.tracing import trace_context


def _log_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.mark.parametrize(
    "key",
    [
        "token",
        "telegram_bot_token",
        "TELEGRAM_WEBHOOK_SECRET",
        "x-telegram-bot-api-secret-token",
        "anthropic_api_key",
        "apikey",
        "password",
        "admin_password_hash",
        "user_id_pepper",
        "Authorization",
        "cookie",
        "x_hub_signature",
        "X-Hub-Signature-256",
        "key",
        "secret_key",
        "private_key",
        "text",
        "caption",
        "transcript",
        "ocr_text",
        "raw_body",
        "message_text",
        "user_text",
        "forwarded_text",
        "claims",
    ],
)
def test_sensitive_keys(key: str) -> None:
    assert is_sensitive_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "event",
        "trace_id",
        "job_id",
        "input_tokens",
        "token_count",
        "duration_ms",
        "status",
        "cache_key",
        "text_length",
        "context",
    ],
)
def test_non_sensitive_keys(key: str) -> None:
    assert not is_sensitive_key(key)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "postgresql+asyncpg://scambot:hunter2@db:5432/scambot",
            f"postgresql+asyncpg://{REDACTED}@db:5432/scambot",
        ),
        ("redis://:pw@redis:6379/0", f"redis://{REDACTED}@redis:6379/0"),
        ("postgresql://user:p@ss@db/scambot", f"postgresql://{REDACTED}@db/scambot"),
        ("bot 1234567890:AAH9xYzAbCdEfGhIjKlMnOpQrStUvWxYz12 failed", f"bot {REDACTED} failed"),
        (
            "https://api.telegram.org/bot1234567890:AAH9xYzAbCdEfGhIjKlMnOpQrStUvWxYz12/getMe",
            f"https://api.telegram.org/bot{REDACTED}/getMe",
        ),
        ("key sk-ant-api03-abcdefghijklmnop", f"key {REDACTED}"),
        ("google AIzaSyA1234567890abcdefghijklmnopqrstuv", f"google {REDACTED}"),
        (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search?query=x&key=k-123",
            f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query=x&key={REDACTED}",
        ),
        ("postgresql://u@db/x?password=hunter2", f"postgresql://u@db/x?password={REDACTED}"),
        (
            "https://api.sightengine.com/1.0/check.json?api_user=123&api_secret=s3cr3t&url=x",
            f"https://api.sightengine.com/1.0/check.json?api_user=123&api_secret={REDACTED}&url=x",
        ),
        (
            "/webhooks/whatsapp?hub.mode=subscribe&hub.verify_token=v-123&hub.challenge=42",
            f"/webhooks/whatsapp?hub.mode=subscribe&hub.verify_token={REDACTED}&hub.challenge=42",
        ),
        ("https://x/y?monkey=banana&cache=1", "https://x/y?monkey=banana&cache=1"),
        ("Authorization: Bearer abc.def.ghi", f"Authorization: Bearer {REDACTED}"),
        ("redis://redis:6379/0", "redis://redis:6379/0"),
        ("https://example.com:8443/a?user=a@b.com", "https://example.com:8443/a?user=a@b.com"),
    ],
)
def test_scrub_string(raw: str, expected: str) -> None:
    assert scrub_string(raw) == expected


@pytest.mark.parametrize("raw", ["a" * 200_000, "http://a:" + "b" * 200_000])
def test_scrub_string_is_fast_on_long_values(raw: str) -> None:
    # A quadratic pattern takes minutes here; a linear one takes milliseconds.
    started = time.perf_counter()
    scrub_string(raw)
    assert time.perf_counter() - started < 1.0


class _Model(BaseModel):
    job_id: str
    text: str
    token: SecretStr


@dataclasses.dataclass
class _Record:
    job_id: str
    caption: str


class _Opaque:
    def __repr__(self) -> str:
        return "Opaque(url='redis://u:p@h/0')"


class _SelfDescribing:
    def __structlog__(self) -> dict[str, str]:
        return {"text": "hello", "url": "redis://u:p@h/0"}


def test_redact_processor_handles_objects() -> None:
    event = {
        "event": "x",
        "job": _Model(job_id="telegram:1", text="hello", token=SecretStr("s")),
        "record": _Record(job_id="telegram:2", caption="hello"),
        "payload": b"redis://u:p@h/0",
        "other": _Opaque(),
        "described": _SelfDescribing(),
        "error": RuntimeError("redis://u:p@h/0"),
        "flag": True,
        "nothing": None,
    }

    result = redact_processor(None, "info", event)

    assert result == {
        "event": "x",
        "job": {"job_id": "telegram:1", "text": REDACTED, "token": REDACTED},
        "record": {"job_id": "telegram:2", "caption": REDACTED},
        "payload": f"b'redis://{REDACTED}@h/0'",
        "other": f"Opaque(url='redis://{REDACTED}@h/0')",
        "described": {"text": REDACTED, "url": f"redis://{REDACTED}@h/0"},
        "error": f"RuntimeError('redis://{REDACTED}@h/0')",
        "flag": True,
        "nothing": None,
    }


def test_redact_processor_handles_nested_values() -> None:
    event = {
        "event": "vendor call failed",
        "telegram_bot_token": "123:abc",
        "headers": {"Authorization": "Bearer x", "Accept": "application/json"},
        "items": [{"text": "hello"}, "redis://u:p@h/0"],
        "pepper": SecretStr("s"),
        "secret_object": SecretStr("s"),
        "duration_ms": 12,
    }

    result = redact_processor(None, "info", event)

    assert result == {
        "event": "vendor call failed",
        "telegram_bot_token": REDACTED,
        "headers": {"Authorization": REDACTED, "Accept": "application/json"},
        "items": [{"text": REDACTED}, f"redis://{REDACTED}@h/0"],
        "pepper": REDACTED,
        "secret_object": REDACTED,
        "duration_ms": 12,
    }


def test_logs_are_json_with_level_timestamp_and_trace_id() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    with trace_context("trace-123"):
        get_logger().info("job received", job_id="telegram:1", kind="text")

    [line] = _log_lines(stream)
    assert line["event"] == "job received"
    assert line["level"] == "info"
    assert line["trace_id"] == "trace-123"
    assert line["job_id"] == "telegram:1"
    assert "timestamp" in line


def test_message_text_and_secrets_never_reach_the_output() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    get_logger().info(
        "intake",
        text="Your electricity will be cut tonight",
        telegram_bot_token="1234567890:AAH9xYzAbCdEfGhIjKlMnOpQrStUvWxYz12",
        database_url="postgresql+asyncpg://scambot:hunter2@db/scambot",
    )

    output = stream.getvalue()
    assert "electricity" not in output
    assert "AAH9xYz" not in output
    assert "hunter2" not in output
    [line] = _log_lines(stream)
    assert line["text"] == REDACTED


def test_level_filtering() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    logger = get_logger()

    logger.info("hidden")
    logger.warning("shown")

    assert [line["event"] for line in _log_lines(stream)] == ["shown"]


def test_numeric_level_is_accepted() -> None:
    stream = io.StringIO()
    configure_logging(logging.ERROR, stream=stream)

    get_logger().warning("hidden")
    get_logger().error("shown")

    assert [line["event"] for line in _log_lines(stream)] == ["shown"]


@pytest.mark.parametrize("level", ["LOUD", 15, True])
def test_unknown_level_is_rejected(level: str | int) -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level)


def test_exceptions_are_rendered_without_local_variables() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    def fail() -> None:
        # Neither the name nor the value looks secret, so only show_locals=False hides it.
        local_note = "local-variable-marker"
        raise RuntimeError("boom" + local_note[:0])

    try:
        fail()
    except RuntimeError:
        get_logger().exception("checker crashed")

    output = stream.getvalue()
    assert "boom" in output
    assert "local-variable-marker" not in output


def test_http_client_loggers_are_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.NOTSET)

    configure_logging("DEBUG")

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
