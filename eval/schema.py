"""Evaluation items: the dataset row model and its JSONL / CSV loaders."""

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scambot.core.contracts import VerdictLabel

Kind = Literal["text", "image", "video", "audio", "voice"]
Suite = Literal[
    "scam", "factcheck", "old_media", "unverifiable", "injection", "genuine", "deepfake", "voice"
]
Language = Literal["en", "hi", "hinglish"]
Split = Literal["dev", "test"]

PLACEHOLDER: Final = "SYNTHETIC-PLACEHOLDER"
_ID_RE: Final = re.compile(r"^[a-z0-9_-]{3,40}$")
_DEVANAGARI_RE: Final = re.compile(r"[ऀ-ॿ]")
_LATIN_RE: Final = re.compile(r"[A-Za-z]")
_DEVANAGARI_RATIO: Final = 0.3

# Targets for a real (not seed) dataset, used only for warnings.
QUOTAS: Final[dict[str, int]] = {
    "old_media": 20,
    "scam": 100,
    "factcheck": 30,
    "unverifiable": 30,
    "injection": 20,
}


class DatasetError(ValueError):
    """A dataset file or row is not valid."""


class Item(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Kind
    text: str = ""
    media_path: str | None = None
    media_url: str | None = None
    suite: Suite
    expected_label: VerdictLabel
    expected_source: str | None = None
    category: str = ""
    language: Language
    split: Split
    source_note: str = Field(min_length=1)
    synthetic: bool = False
    canary: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _ID_RE.match(self.id):
            msg = f"bad id {self.id!r}"
            raise ValueError(msg)
        if self.kind != "text" and not (self.media_path or self.media_url):
            msg = f"{self.id}: media items need media_path or media_url"
            raise ValueError(msg)
        if self.kind == "text" and not self.text.strip():
            msg = f"{self.id}: text items need text"
            raise ValueError(msg)
        if self.source_note.startswith("SYNTHETIC") and not self.synthetic:
            msg = f"{self.id}: SYNTHETIC source_note requires synthetic=true"
            raise ValueError(msg)
        if self.canary is not None and self.suite != "injection":
            msg = f"{self.id}: canary is only allowed in the injection suite"
            raise ValueError(msg)
        ratio = devanagari_ratio(self.text)
        if self.text and self.language == "hi" and ratio == 0:
            msg = f"{self.id}: language hi but no Devanagari text"
            raise ValueError(msg)
        if self.text and self.language != "hi" and ratio > _DEVANAGARI_RATIO:
            msg = f"{self.id}: mostly Devanagari text but language is {self.language}"
            raise ValueError(msg)
        return self


def devanagari_ratio(text: str) -> float:
    """Share of Devanagari characters among Devanagari and Latin letters."""
    marks = len(_DEVANAGARI_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    return marks / (marks + latin) if marks + latin else 0.0


def _rows(path: Path) -> Iterator[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield {k: (v if v != "" else None) for k, v in row.items() if k}
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path.name}:{number}: not valid JSON"
                raise DatasetError(msg) from exc


def _coerce(row: dict[str, object]) -> dict[str, object]:
    fixed = dict(row)
    if isinstance(fixed.get("synthetic"), str):
        fixed["synthetic"] = str(fixed["synthetic"]).strip().lower() in {"1", "true", "yes"}
    if fixed.get("text") is None:
        fixed["text"] = ""
    for key in ("category",):
        if fixed.get(key) is None:
            fixed[key] = ""
    return fixed


def load_items(
    path: Path, *, check_media: bool = False, forbid_placeholders: bool = False
) -> list[Item]:
    """Load one JSONL or CSV file. Media paths are relative to the file's folder."""
    items: list[Item] = []
    seen: set[str] = set()
    for row in _rows(path):
        try:
            item = Item.model_validate(_coerce(row))
        except ValueError as exc:
            msg = f"{path.name}: {exc}"
            raise DatasetError(msg) from exc
        if item.id in seen:
            msg = f"{path.name}: duplicate id {item.id}"
            raise DatasetError(msg)
        seen.add(item.id)
        if forbid_placeholders and item.source_note.startswith(PLACEHOLDER):
            msg = f"{path.name}: {item.id} is a placeholder row, not real data"
            raise DatasetError(msg)
        if check_media and item.media_path and not (path.parent / item.media_path).is_file():
            msg = f"{path.name}: {item.id} media file {item.media_path} is missing"
            raise DatasetError(msg)
        items.append(item)
    return items


def media_file(item: Item, folder: Path) -> Path | None:
    return folder / item.media_path if item.media_path else None


def dataset_files(directory: Path) -> list[Path]:
    """Dataset files directly inside ``directory`` (not templates or article records)."""
    return sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in {".jsonl", ".csv"}
        and not p.name.endswith(".template.jsonl")
        and "articles" not in p.name
    )


def load_dir(directory: Path, **kwargs: bool) -> list[Item]:
    """Load every dataset file directly inside ``directory``."""
    items: list[Item] = []
    seen: set[str] = set()
    for path in dataset_files(directory):
        for item in load_items(path, **kwargs):
            if item.id in seen:
                msg = f"duplicate id {item.id} across files"
                raise DatasetError(msg)
            seen.add(item.id)
            items.append(item)
    return items


def load_with_folders(paths: Sequence[Path]) -> tuple[list[Item], dict[str, Path]]:
    """Files or folders; also the folder each item's media paths are relative to.

    A path with a ``real`` folder in it rejects placeholder rows (the templates).
    """
    items: list[Item] = []
    folders: dict[str, Path] = {}
    for path in paths:
        guard = "real" in path.resolve().parts
        files = dataset_files(path) if path.is_dir() else [path]
        for file in files:
            for item in load_items(file, forbid_placeholders=guard):
                if item.id in folders:
                    msg = f"duplicate id {item.id} across inputs"
                    raise DatasetError(msg)
                folders[item.id] = file.parent
                items.append(item)
    return items, folders


def iter_split(items: Iterable[Item], split: str) -> Iterator[Item]:
    for item in items:
        if split in ("all", item.split):
            yield item


def split_counts(items: Iterable[Item]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for item in items:
        counts.setdefault(item.suite, Counter())[item.split] += 1
    return {suite: dict(c) for suite, c in sorted(counts.items())}


def validate_dataset_report(items: Sequence[Item]) -> dict[str, object]:
    """Per-suite counts plus warnings for unmet real-set quotas (warnings are not errors)."""
    per_suite = Counter[str](i.suite for i in items)
    warnings: list[str] = []
    for suite, quota in QUOTAS.items():
        have = per_suite[suite]
        if have < quota:
            warnings.append(f"{suite}: {have} of {quota} items")
    hindi = sum(1 for i in items if i.suite == "factcheck" and i.language == "hi")
    if per_suite.get("factcheck") and hindi < 10:
        warnings.append(f"factcheck: {hindi} of 10 Hindi items")
    return {"counts": dict(per_suite), "splits": split_counts(items), "warnings": warnings}
