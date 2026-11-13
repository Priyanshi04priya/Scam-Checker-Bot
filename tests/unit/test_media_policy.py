"""Tests for the shared media policy."""

import pytest

from scambot.core.contracts import JobKind, MediaMeta
from scambot.core.media import MAX_MEDIA_BYTES, declared_problem, kind_for_mime, normalise_mime


def meta(**fields: object) -> MediaMeta:
    return MediaMeta.model_validate({"source": "document", **fields})


def test_oversized_declared_file_is_too_large() -> None:
    assert declared_problem("voice", meta(file_size=MAX_MEDIA_BYTES + 1)) == "too_large"
    assert declared_problem("voice", meta(file_size=MAX_MEDIA_BYTES)) is None


@pytest.mark.parametrize(
    ("kind", "mime", "problem"),
    [
        ("image", "image/png", None),
        ("image", "image/gif", "unsupported_type"),
        ("image", "application/x-msdownload", "unsupported_type"),
        ("video", "video/mp4; codecs=avc1", None),
        ("video", "audio/ogg", "unsupported_type"),
        ("voice", "audio/ogg", None),
        ("audio", "video/mp4", None),
        ("text", "image/png", "unsupported_type"),
    ],
)
def test_declared_type_is_checked_per_kind(kind: JobKind, mime: str, problem: str | None) -> None:
    assert declared_problem(kind, meta(mime_type=mime)) == problem


def test_nothing_declared_cannot_be_prechecked() -> None:
    assert declared_problem("image", None) is None
    assert declared_problem("image", meta()) is None


@pytest.mark.parametrize(
    ("mime", "kind"),
    [
        ("image/jpeg", "image"),
        ("video/mp4; codecs=avc1", "video"),
        ("AUDIO/MPEG", "audio"),
        ("application/pdf", None),
        ("", None),
        (None, None),
    ],
)
def test_kind_for_mime(mime: str | None, kind: str | None) -> None:
    assert kind_for_mime(mime) == kind


def test_normalise_mime() -> None:
    assert normalise_mime(" Video/MP4 ; codecs=x") == "video/mp4"
