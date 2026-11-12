"""Tests for the shared data contracts."""

import pickle
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from scambot.core.contracts import CheckerResult, Command, Evidence, Job, Verdict


def job(**overrides: Any) -> Job:
    fields: dict[str, Any] = {
        "job_id": "telegram:1",
        "trace_id": "trace",
        "channel": "telegram",
        "chat_id": "424242",
        "user_hash": "abc",
        "kind": "text",
        "text": "secret message body",
        "media_ref": "file-id-xyz",
        "forward_meta": {"name": "Ravi"},
        "received_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    return Job(**(fields | overrides))


def test_job_repr_hides_user_content() -> None:
    shown = repr(job()) + str(job())

    for hidden in ("secret message body", "424242", "file-id-xyz", "Ravi"):
        assert hidden not in shown
    assert "telegram:1" in shown


def test_job_survives_the_queue_round_trip() -> None:
    original = job(ack_message_id="9")

    assert pickle.loads(pickle.dumps(original)) == original  # noqa: S301
    assert Job.model_validate_json(original.model_dump_json()) == original


def test_job_is_immutable() -> None:
    with pytest.raises(ValidationError):
        job().text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"channel": "sms"},
        {"kind": "sticker"},
        {"job_id": ""},
        {"received_at": datetime(2026, 1, 1)},  # noqa: DTZ001
        {"unexpected": 1},
    ],
)
def test_job_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        job(**overrides)


@pytest.mark.parametrize("name", ["Start", "", "a" * 33, "start now"])
def test_command_name_must_be_normalised(name: str) -> None:
    with pytest.raises(ValidationError):
        Command(
            command_id="telegram:1",
            trace_id="t",
            channel="telegram",
            chat_id="1",
            user_hash="h",
            name=name,
        )


def test_checker_result_bounds() -> None:
    result = CheckerResult(
        checker="analyze_scam_patterns",
        status="ok",
        signal="scam",
        confidence=0.9,
        evidence=[Evidence(summary="asks for OTP")],
        duration_ms=12,
    )
    assert result.details == {}
    with pytest.raises(ValidationError):
        result.model_validate(result.model_dump() | {"confidence": 1.5})


def test_verdict_defaults_and_score_range() -> None:
    verdict = Verdict(label="could_not_verify", band="unknown", score=50)

    assert verdict.contributing == []
    assert verdict.unavailable == []
    assert verdict.source_confirmed is False
    with pytest.raises(ValidationError):
        Verdict(label="no_issues", band="low", score=101)
