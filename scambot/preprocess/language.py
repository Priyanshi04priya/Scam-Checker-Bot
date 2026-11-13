"""Language detection for extracted text."""

import importlib
import importlib.util
import re
import threading
from collections import Counter
from typing import Any, Final, Protocol

# (first code point, last code point, language)
_SCRIPTS: Final = (
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0600, 0x06FF, "ur"),
    (0x0041, 0x005A, "en"),
    (0x0061, 0x007A, "en"),
)
_URL_RE: Final = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+", re.IGNORECASE)
_LINGUA_LANGUAGES: Final = (
    "ENGLISH",
    "HINDI",
    "MARATHI",
    "BENGALI",
    "GUJARATI",
    "PUNJABI",
    "TAMIL",
    "TELUGU",
    "URDU",
)
MIN_LETTERS: Final = 3
MIN_SHARE: Final = 0.5


class LanguageDetector(Protocol):
    def detect(self, text: str) -> str | None:
        """ISO 639-1 code of the language of ``text``, or None when unsure."""
        ...


class ScriptLanguageDetector:
    """Guesses the language from the writing system; Latin text counts as English."""

    def detect(self, text: str) -> str | None:
        counts: Counter[str] = Counter()
        for char in _URL_RE.sub(" ", text):
            code = ord(char)
            for low, high, language in _SCRIPTS:
                if low <= code <= high:
                    counts[language] += 1
                    break
        total = sum(counts.values())
        if total < MIN_LETTERS:
            return None
        language, count = counts.most_common(1)[0]
        return language if count / total >= MIN_SHARE else None


class LinguaLanguageDetector:
    """Statistical detector from the optional ``media`` group; falls back to scripts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detector: Any = None
        self._fallback = ScriptLanguageDetector()

    def _build(self) -> Any:
        with self._lock:
            if self._detector is None:
                lingua = importlib.import_module("lingua")
                languages = [getattr(lingua.Language, name) for name in _LINGUA_LANGUAGES]
                self._detector = (
                    lingua.LanguageDetectorBuilder.from_languages(*languages)
                    .with_minimum_relative_distance(0.1)
                    .build()
                )
            return self._detector

    def detect(self, text: str) -> str | None:
        found = self._build().detect_language_of(text)
        if found is None:
            return self._fallback.detect(text)
        return str(found.iso_code_639_1.name).lower()


def create_language_detector() -> LanguageDetector:
    """Lingua when installed, otherwise the script-based detector."""
    if importlib.util.find_spec("lingua") is not None:
        return LinguaLanguageDetector()
    return ScriptLanguageDetector()
