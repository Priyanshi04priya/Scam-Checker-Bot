"""Media policy shared by the channel adapters, intake and file validation."""

from collections.abc import Mapping
from typing import Final

from scambot.core.contracts import JobKind, MediaMeta, RejectReason

# The Telegram Bot API serves files up to this size. It is a security boundary, not a tunable.
MAX_MEDIA_BYTES: Final = 20 * 1024 * 1024

# No GIF, HEIC or SVG: Pillow cannot read HEIC, and Telegram sends GIFs as mp4 animations.
IMAGE_MIME: Final = frozenset({"image/jpeg", "image/png", "image/webp"})
VIDEO_MIME: Final = frozenset(
    {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/3gpp"}
)
AUDIO_MIME: Final = frozenset(
    {
        "audio/ogg",
        "audio/opus",
        "audio/mpeg",
        "audio/mp3",
        "audio/m4a",
        "audio/x-opus+ogg",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "audio/wav",
        "audio/x-wav",
        "audio/flac",
        "audio/x-flac",
        "audio/amr",
        "audio/webm",
    }
)
# Audio-only mp4, webm and 3gp containers are recognised as video from their header.
_AUDIO_CONTAINERS: Final = frozenset({"video/mp4", "video/webm", "video/3gpp"})

ACCEPTED_MIME: Final[Mapping[JobKind, frozenset[str]]] = {
    "text": frozenset(),
    "image": IMAGE_MIME,
    "video": VIDEO_MIME,
    "audio": AUDIO_MIME | _AUDIO_CONTAINERS,
    "voice": AUDIO_MIME | _AUDIO_CONTAINERS,
}


def normalise_mime(mime: str) -> str:
    """Lower-case media type without parameters (``Video/MP4; codecs=x`` -> ``video/mp4``)."""
    return mime.split(";", 1)[0].strip().lower()


def kind_for_mime(mime: str | None) -> JobKind | None:
    """The job kind for a declared media type, or None when the bot does not accept it."""
    if not mime:
        return None
    mime = normalise_mime(mime)
    if mime in IMAGE_MIME:
        return "image"
    if mime in VIDEO_MIME:
        return "video"
    if mime in AUDIO_MIME:
        return "audio"
    return None


def declared_problem(kind: JobKind, meta: MediaMeta | None) -> RejectReason | None:
    """A reason to refuse a file from what the platform declares, before downloading it."""
    if meta is None:
        return None
    if meta.file_size is not None and meta.file_size > MAX_MEDIA_BYTES:
        return "too_large"
    if meta.mime_type and normalise_mime(meta.mime_type) not in ACCEPTED_MIME[kind]:
        return "unsupported_type"
    return None


class MediaRejected(Exception):  # noqa: N818 - a domain outcome, not an error condition
    """The file cannot be checked; ``reason`` selects the message for the user."""

    def __init__(self, reason: RejectReason) -> None:
        super().__init__(reason)
        self.reason: RejectReason = reason


class MediaUnavailable(Exception):  # noqa: N818
    """A transient failure while fetching the file; the message is a short fixed code."""
