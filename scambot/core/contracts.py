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
RatingPolarity = Literal["false", "misleading", "mixed", "true", "unrated"]
RejectReason = Literal["too_large", "unsupported_type", "unreadable"]
MediaSource = Literal["photo", "document", "video", "video_note", "animation", "audio", "voice"]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MediaMeta(_Contract):
    """What the platform declares about a file before it is downloaded."""

    source: MediaSource
    file_size: int | None = Field(default=None, ge=0)
    mime_type: str | None = None
    duration_s: int | None = Field(default=None, ge=0)


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
    media: MediaMeta | None = None
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


class Rejection(_Contract):
    """An update refused at intake: answered there and never queued."""

    rejection_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    channel: Channel
    chat_id: str = Field(min_length=1, repr=False)
    user_hash: str = Field(min_length=1)
    reason: RejectReason
    preferred_lang: str | None = None


class MediaFeatures(_Contract):
    """What preprocessing learned from a media file. No file name is ever kept."""

    mime_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phash: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    keyframe_phashes: list[str] = Field(default_factory=list)
    keyframe_names: list[str] = Field(default_factory=list)
    ocr_text: str = Field(default="", repr=False)
    transcript: str = Field(default="", repr=False)
    speech_language: str | None = None
    errors: list[str] = Field(default_factory=list)


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


CheckerName = Literal[
    "analyze_scam_patterns",
    "fact_check_claim",
    "check_provenance",
    "check_caption_context",
    "detect_deepfake",
    "detect_voice_clone",
]
ScamTemplate = Literal[
    "kyc_update",
    "lottery_prize",
    "job_offer",
    "digital_arrest",
    "electricity_disconnection",
    "courier_customs",
    "investment_doubling",
    "official_impersonation",
]
LexiconSignal = Literal["urgency", "threat", "payment_request", "otp_request"]
EntitySignal = Literal["phone_number", "upi_id", "shortened_url", "lookalike_domain", "unsafe_url"]
ScamSignal = ScamTemplate | LexiconSignal | EntitySignal


class UrlEntity(_Contract):
    url: str
    expanded_url: str | None = None


class Entities(_Contract):
    """Links, phone numbers, UPI IDs and e-mail addresses found in a message."""

    urls: list[UrlEntity] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    upi_ids: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)


class Preprocessed(_Contract):
    """The checkable form of a job: its text, the entities in it and its language."""

    kind: JobKind
    text: str = Field(default="", repr=False)
    entities: Entities = Field(default_factory=Entities, repr=False)
    caption: str = Field(default="", repr=False)
    language: str | None = None
    has_media: bool = False
    media: MediaFeatures | None = Field(default=None, repr=False)


class ExecutedCheck(_Contract):
    """A checker result together with the hash of the arguments it was run on."""

    result: CheckerResult
    input_hash: str
