"""Platform-independent data contracts shared by every package."""

import datetime as dt
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

Channel = Literal["telegram", "whatsapp"]
JobKind = Literal["text", "image", "video", "audio", "voice"]
CheckerStatus = Literal["ok", "inconclusive", "error"]
Signal = Literal["old_media", "out_of_context", "debunked", "scam", "likely_synthetic", "none"]
VerdictLabel = Literal[
    "false_misleading",
    "likely_scam",
    "old_or_out_of_context",
    "possibly_ai",
    "no_issues",
    "could_not_verify",
]
Band = Literal["high", "medium", "low", "unknown"]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Job(_Contract):
    """One message to check: the only thing a channel adapter hands to the pipeline.

    Fields that hold user content or chat identifiers are left out of ``repr()`` so that queue
    and error logs never show them.
    """

    job_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    channel: Channel
    chat_id: str = Field(min_length=1, repr=False)
    user_hash: str = Field(min_length=1)
    kind: JobKind
    text: str | None = Field(default=None, repr=False)
    media_ref: str | None = Field(default=None, repr=False)
    forward_meta: dict[str, Any] = Field(default_factory=dict, repr=False)
    ack_message_id: str | None = None
    preferred_lang: str | None = None
    received_at: AwareDatetime


class Command(_Contract):
    """A bot command (``/start``, ``/help``, ...) answered at intake, never queued."""

    command_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    channel: Channel
    chat_id: str = Field(min_length=1, repr=False)
    user_hash: str = Field(min_length=1)
    name: str = Field(pattern=r"^[a-z0-9_]{1,32}$")
    args: str = Field(default="", repr=False)
    preferred_lang: str | None = None


class Evidence(_Contract):
    summary: str
    url: str | None = None
    source: str | None = None
    date: dt.date | None = None


class CheckerResult(_Contract):
    checker: str
    status: CheckerStatus
    signal: Signal
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    duration_ms: int = Field(ge=0)
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class Verdict(_Contract):
    label: VerdictLabel
    band: Band
    score: int = Field(ge=0, le=100)
    deciding_checker: str | None = None
    contributing: list[tuple[str, str, str]] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)
    source_confirmed: bool = False
