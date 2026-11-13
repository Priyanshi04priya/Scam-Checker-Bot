"""Tests for OCR."""

import shutil
from pathlib import Path
from typing import Any

import pytesseract
import pytest
from PIL import Image

from scambot.core.config import AppConfig
from scambot.core.settings import Settings
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.ocr import NoopOcr, TesseractOcr, create_ocr
from tests.media import HINDI_NOTICE


def tesseract_has_hindi() -> bool:
    if shutil.which("tesseract") is None:
        return False
    return "hin" in pytesseract.get_languages(config="")


def data(*words: tuple[str, int, int]) -> dict[str, list[Any]]:
    """Word, confidence and line number per word."""
    return {
        "text": [w for w, _, _ in words],
        "conf": [c for _, c, _ in words],
        "block_num": [1] * len(words),
        "par_num": [1] * len(words),
        "line_num": [line for _, _, line in words],
    }


@pytest.fixture
def small_image(tmp_path: Path) -> Path:
    path = tmp_path / "original"
    Image.new("RGB", (300, 100), "white").save(path, format="PNG")
    return path


def test_low_confidence_words_are_dropped_and_lines_rebuilt(
    app_config: AppConfig, small_image: Path
) -> None:
    seen: list[Image.Image] = []
    kwargs: dict[str, Any] = {}

    def fake(image: Image.Image, **options: Any) -> dict[str, list[Any]]:
        seen.append(image)
        kwargs.update(options)
        return data(
            ("Your", 95, 1), ("KYC", 91, 1), ("zzz", 12, 1), ("", -1, 1), ("expires", 88, 2)
        )

    result = TesseractOcr(app_config.thresholds.media.ocr, image_to_data=fake).extract(small_image)

    assert result.text == "Your KYC\nexpires"
    assert result.language is None
    assert seen[0].mode == "L"
    assert seen[0].size == (600, 200)  # shorter side below 1000 px: scaled 2x
    assert kwargs["lang"] == "eng+hin"
    assert kwargs["timeout"] == 15


def test_too_little_text_is_nothing(app_config: AppConfig, small_image: Path) -> None:
    engine = TesseractOcr(
        app_config.thresholds.media.ocr, image_to_data=lambda *_a, **_k: data(("-", 90, 1))
    )

    assert engine.extract(small_image).text == ""


def test_large_images_are_scaled_down(app_config: AppConfig, tmp_path: Path) -> None:
    path = tmp_path / "big"
    Image.new("RGB", (4000, 2000), "white").save(path, format="PNG")
    sizes: list[tuple[int, int]] = []

    def fake(image: Image.Image, **_options: Any) -> dict[str, list[Any]]:
        sizes.append(image.size)
        return data()

    TesseractOcr(app_config.thresholds.media.ocr, image_to_data=fake).extract(path)

    assert sizes == [(3000, 1500)]


def test_tesseract_timeout_is_reported(app_config: AppConfig, small_image: Path) -> None:
    def fake(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
        msg = "Tesseract process timeout"
        raise RuntimeError(msg)

    engine = TesseractOcr(app_config.thresholds.media.ocr, image_to_data=fake)

    with pytest.raises(MediaToolError) as raised:
        engine.extract(small_image)
    assert raised.value.code == "timeout"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (pytesseract.TesseractNotFoundError(), "unavailable"),
        (pytesseract.TesseractError(1, "boom"), "failed"),
        (RuntimeError("other"), "failed"),
    ],
)
def test_tesseract_errors_map_to_codes(
    app_config: AppConfig, small_image: Path, error: Exception, code: str
) -> None:
    def fake(*_a: Any, **_k: Any) -> dict[str, list[Any]]:
        raise error

    engine = TesseractOcr(app_config.thresholds.media.ocr, image_to_data=fake)

    with pytest.raises(MediaToolError) as raised:
        engine.extract(small_image)
    assert raised.value.code == code


def test_unreadable_image(app_config: AppConfig, tmp_path: Path) -> None:
    path = tmp_path / "bad"
    path.write_bytes(b"not an image")

    with pytest.raises(MediaToolError) as raised:
        TesseractOcr(app_config.thresholds.media.ocr, image_to_data=lambda *_a, **_k: {}).extract(
            path
        )
    assert raised.value.code == "bad_image"


@pytest.mark.skipif(not tesseract_has_hindi(), reason="tesseract with Hindi data is not installed")
def test_real_tesseract_reads_the_hindi_notice(app_config: AppConfig) -> None:
    result = TesseractOcr(app_config.thresholds.media.ocr).extract(HINDI_NOTICE)

    assert "बिजली" in result.text
    assert "9876543210" in result.text


def test_ocr_engine_none_is_a_noop(app_config: AppConfig) -> None:
    engine = create_ocr(Settings(ocr_engine="none"), app_config.thresholds.media.ocr)

    assert isinstance(engine, NoopOcr)
    with pytest.raises(MediaToolError):
        engine.extract(Path("x"))


def test_missing_binary_or_language_gives_a_noop(
    app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = app_config.thresholds.media.ocr
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert isinstance(create_ocr(Settings(), cfg), NoopOcr)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng"])
    assert isinstance(create_ocr(Settings(), cfg), NoopOcr)

    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng", "hin"])
    assert isinstance(create_ocr(Settings(), cfg), TesseractOcr)

    def broken(config: str = "") -> list[str]:
        raise pytesseract.TesseractNotFoundError

    monkeypatch.setattr(pytesseract, "get_languages", broken)
    assert isinstance(create_ocr(Settings(), cfg), NoopOcr)
