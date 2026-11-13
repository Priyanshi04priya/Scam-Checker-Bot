"""Tests for language detection."""

import importlib.util

import pytest

from scambot.preprocess import language
from scambot.preprocess.language import ScriptLanguageDetector, create_language_detector
from tests.samples import ELECTRICITY_EN, ELECTRICITY_HI


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (ELECTRICITY_HI, "hi"),
        (ELECTRICITY_EN, "en"),
        ("மின்சாரம் இன்று இரவு துண்டிக்கப்படும்", "ta"),
        ("ok 123", None),
        ("", None),
        (f"{ELECTRICITY_HI} https://example.com/verylongpathwithlatinletters", "hi"),
    ],
)
def test_script_detector(text: str, expected: str | None) -> None:
    assert ScriptLanguageDetector().detect(text) == expected


def test_script_detector_is_the_default_without_lingua(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert isinstance(create_language_detector(), ScriptLanguageDetector)


def test_lingua_detector() -> None:
    pytest.importorskip("lingua")
    detector = language.LinguaLanguageDetector()

    assert detector.detect(ELECTRICITY_HI) == "hi"
    assert detector.detect(ELECTRICITY_EN) == "en"
    assert detector.detect("12 34") == detector.detect("12 34")  # undecidable text does not raise


def test_lingua_is_chosen_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())

    assert isinstance(create_language_detector(), language.LinguaLanguageDetector)
