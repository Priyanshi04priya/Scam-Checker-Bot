"""Content fingerprints. Only hashes of messages are stored, never the text itself."""

import hashlib
import re

_SPACE_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lower-case with runs of whitespace collapsed, so trivial edits hash the same."""
    return _SPACE_RE.sub(" ", text.lower()).strip()


def text_sha256(text: str) -> str | None:
    """SHA-256 of the normalised text, or None when there is no text."""
    normalised = normalise_text(text)
    if not normalised:
        return None
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
