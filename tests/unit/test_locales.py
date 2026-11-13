"""Tests for the locale files and loader."""

from pathlib import Path
from typing import get_args

import pytest
import yaml

from scambot.core.contracts import CheckerName, ScamSignal
from scambot.core.locales import (
    CHECKS_PLACEHOLDER,
    LOCALES_DIR,
    SIGNALS_PLACEHOLDER,
    Locales,
    LocaleTexts,
    load_locales,
)


def test_english_and_hindi_are_loaded(locales: Locales) -> None:
    assert locales.languages == {"en", "hi"}
    assert locales.get("en").checking == "Checking…"
    assert locales.get("hi").checking != locales.get("en").checking


@pytest.mark.parametrize(
    ("language", "resolved"),
    [
        ("hi", "hi"),
        ("hi-IN", "hi"),
        ("HI_in", "hi"),
        ("en-GB", "en"),
        ("ta", "en"),
        (None, "en"),
        ("", "en"),
    ],
)
def test_resolve(locales: Locales, language: str | None, resolved: str) -> None:
    assert locales.resolve(language) == resolved


def test_every_language_has_the_same_keys() -> None:
    def keys(data: object, prefix: str = "") -> set[str]:
        if isinstance(data, dict):
            return {k for key, value in data.items() for k in keys(value, f"{prefix}{key}.")}
        return {prefix}

    files = sorted(LOCALES_DIR.glob("*.yaml"))
    assert [path.stem for path in files] == ["en", "hi"]
    en, hi = (keys(yaml.safe_load(path.read_text(encoding="utf-8"))) for path in files)
    assert en == hi


def test_hindi_texts_are_in_devanagari(locales: Locales) -> None:
    commands = locales.get("hi").commands
    for text in (commands.start, commands.help, commands.privacy, commands.unknown):
        assert any("ऀ" <= char <= "ॿ" for char in text)


def test_start_help_privacy_cover_what_users_must_be_told(locales: Locales) -> None:
    commands = locales.get("en").commands
    assert "90 days" in commands.start
    assert "20 MB" in commands.help
    assert "/privacy" in commands.start
    assert "1 hour" in commands.privacy
    assert "90 days" in commands.privacy


def _write(directory: Path, name: str, data: object) -> None:
    (directory / f"{name}.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
    )


def _english() -> dict[str, object]:
    return yaml.safe_load((LOCALES_DIR / "en.yaml").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_missing_key_fails_loading(tmp_path: Path) -> None:
    data = _english()
    del data["checking"]
    _write(tmp_path, "en", data)

    with pytest.raises(ValueError, match=r"en\.yaml"):
        load_locales(tmp_path)


def test_unknown_key_fails_loading(tmp_path: Path) -> None:
    _write(tmp_path, "en", _english() | {"surprise": "x"})

    with pytest.raises(ValueError, match=r"en\.yaml"):
        load_locales(tmp_path)


def test_empty_text_fails_loading(tmp_path: Path) -> None:
    _write(tmp_path, "en", _english() | {"checking": ""})

    with pytest.raises(ValueError, match=r"en\.yaml"):
        load_locales(tmp_path)


def test_english_is_required(tmp_path: Path) -> None:
    _write(tmp_path, "hi", _english())

    with pytest.raises(ValueError, match="'en' locale is required"):
        load_locales(tmp_path)


def test_locale_model_matches_file() -> None:
    assert LocaleTexts.model_validate(_english()).checking == "Checking…"


@pytest.mark.parametrize("language", ["en", "hi"])
def test_verdict_texts_cover_every_signal_and_checker(locales: Locales, language: str) -> None:
    verdict = locales.get(language).verdict
    signals = {name for alias in get_args(ScamSignal) for name in get_args(alias)}
    assert set(verdict.signals.model_dump()) == signals
    assert set(verdict.checkers.model_dump()) == set(get_args(CheckerName))
    assert SIGNALS_PLACEHOLDER in verdict.explanations.likely_scam
    assert CHECKS_PLACEHOLDER in verdict.unavailable
    assert locales.get(language).timeout


def _with_verdict(change: dict[str, object]) -> dict[str, object]:
    data = _english()
    verdict = dict(data["verdict"])  # type: ignore[call-overload]
    for key, value in change.items():
        section, _, name = key.partition(".")
        if name:
            verdict[section] = dict(verdict[section]) | {name: value}
        else:
            verdict[section] = value
    return data | {"verdict": verdict}


@pytest.mark.parametrize(
    "change",
    [
        {"explanations.likely_scam": "No placeholder here."},
        {"unavailable": "Some checks could not run."},
        {"sources": "Sources {x}"},
        {"signals.urgency": "pressure {to} act"},
    ],
)
def test_bad_placeholders_fail_loading(tmp_path: Path, change: dict[str, object]) -> None:
    _write(tmp_path, "en", _with_verdict(change))

    with pytest.raises(ValueError, match=r"en\.yaml"):
        load_locales(tmp_path)


@pytest.mark.parametrize("language", ["en", "hi"])
def test_rejection_texts_exist(locales: Locales, language: str) -> None:
    rejected = locales.get(language).rejected

    assert "20 MB" in rejected.too_large
    assert rejected.unsupported_type
    assert rejected.unreadable


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rate_limited", "Too many checks."),
        ("rate_limited", "{limit} checks, try in {minutes} {oops}"),
        ("cached", "Earlier result."),
        ("cached", "On {when} {extra}"),
    ],
)
def test_filled_in_texts_need_exactly_their_placeholders(field: str, value: str) -> None:
    data = yaml.safe_load((LOCALES_DIR / "en.yaml").read_text(encoding="utf-8"))
    data[field] = value

    with pytest.raises(ValueError, match=field):
        LocaleTexts.model_validate(data)


def test_the_delete_confirmation_needs_a_count() -> None:
    data = yaml.safe_load((LOCALES_DIR / "en.yaml").read_text(encoding="utf-8"))
    data["commands"]["delete"] = "Done."

    with pytest.raises(ValueError, match=r"commands\.delete"):
        LocaleTexts.model_validate(data)


@pytest.mark.parametrize("language", ["en", "hi"])
def test_button_titles_fit_whatsapp(locales: Locales, language: str) -> None:
    buttons = locales.get(language).buttons

    assert max(len(buttons.details), len(buttons.helpful), len(buttons.wrong)) <= 20
    assert len(buttons.prompt) <= 1024
    assert "update that message" not in locales.get(language).commands.help
