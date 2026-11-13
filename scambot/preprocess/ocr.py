"""Text from images with Tesseract."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output, TesseractError, TesseractNotFoundError

from scambot.core.config import OcrThresholds
from scambot.core.logging import get_logger
from scambot.core.settings import Settings
from scambot.preprocess.base import Extracted, MediaToolError

_log = get_logger(component="ocr")
_UPSCALE: Final = 2

ImageToData = Callable[..., dict[str, list[Any]]]


class OcrEngine(Protocol):
    def extract(self, path: Path) -> Extracted:
        """Read the text in the image at ``path`` (blocking)."""
        ...


class NoopOcr:
    def extract(self, path: Path) -> Extracted:
        raise MediaToolError("unavailable")


class TesseractOcr:
    """Reads Latin and Devanagari text; words Tesseract is unsure about are dropped."""

    def __init__(self, cfg: OcrThresholds, *, image_to_data: ImageToData | None = None) -> None:
        self._cfg = cfg
        self._image_to_data: ImageToData = image_to_data or pytesseract.image_to_data

    def _prepare(self, path: Path) -> Image.Image:
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("L")
        except (OSError, ValueError, SyntaxError):
            raise MediaToolError("bad_image") from None
        if min(image.size) < self._cfg.upscale_below_px:
            image = image.resize(
                (image.width * _UPSCALE, image.height * _UPSCALE), Image.Resampling.LANCZOS
            )
        longest = max(image.size)
        if longest > self._cfg.max_side_px:
            scale = self._cfg.max_side_px / longest
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    def extract(self, path: Path) -> Extracted:
        image = self._prepare(path)
        try:
            data = self._image_to_data(
                image,
                lang=self._cfg.languages,
                config="--psm 3",
                output_type=Output.DICT,
                timeout=self._cfg.timeout_s,
            )
        except TesseractNotFoundError:
            raise MediaToolError("unavailable") from None
        except TesseractError:
            raise MediaToolError("failed") from None
        except RuntimeError as exc:
            raise MediaToolError("timeout" if "timeout" in str(exc).lower() else "failed") from None
        lines: dict[tuple[int, int, int], list[str]] = {}
        for i, word in enumerate(data["text"]):
            if not str(word).strip() or float(data["conf"][i]) < self._cfg.min_word_conf:
                continue
            line = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(line, []).append(str(word).strip())
        text = "\n".join(" ".join(words) for words in lines.values())
        if sum(char.isalnum() for char in text) < self._cfg.min_chars:
            return Extracted("", None)
        return Extracted(text, None)


def create_ocr(settings: Settings, cfg: OcrThresholds) -> OcrEngine:
    """Tesseract when enabled, installed and holding every configured language."""
    if settings.ocr_engine != "tesseract":
        return NoopOcr()
    if shutil.which("tesseract") is None:
        _log.warning("ocr_unavailable", reason="no_binary")
        return NoopOcr()
    try:
        installed = set(pytesseract.get_languages(config=""))
    except (TesseractError, TesseractNotFoundError, RuntimeError):
        _log.warning("ocr_unavailable", reason="languages_unreadable")
        return NoopOcr()
    if not set(cfg.languages.split("+")) <= installed:
        _log.warning("ocr_unavailable", reason="missing_language")
        return NoopOcr()
    return TesseractOcr(cfg)
