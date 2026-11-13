"""Dataset rows: loading, validation and guards."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from eval.schema import (
    DatasetError,
    Item,
    iter_split,
    load_dir,
    load_items,
    load_with_folders,
    split_counts,
    validate_dataset_report,
)


def row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "item-001",
        "kind": "text",
        "text": "hello there",
        "media_path": None,
        "media_url": None,
        "suite": "scam",
        "expected_label": "no_issues",
        "expected_source": None,
        "category": "x",
        "language": "en",
        "split": "dev",
        "source_note": "written for this project",
        "synthetic": False,
    }
    return base | over


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


def test_jsonl_and_csv_load_the_same(tmp_path: Path) -> None:
    rows = [row(), row(id="item-002", text="नमस्ते दोस्त", language="hi", split="test")]
    jsonl = write_jsonl(tmp_path / "a.jsonl", rows)
    csv_path = tmp_path / "a.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
    assert load_items(jsonl) == load_items(csv_path)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "a.jsonl", [row(), row()])
    with pytest.raises(DatasetError, match="duplicate"):
        load_items(path)


def test_duplicate_ids_across_files_rejected(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "a.jsonl", [row()])
    write_jsonl(tmp_path / "b.jsonl", [row()])
    with pytest.raises(DatasetError, match="across files"):
        load_dir(tmp_path)


def test_missing_media_detected(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "a.jsonl", [row(kind="image", media_path="images/a.jpg", text="")]
    )
    assert load_items(path)[0].media_path == "images/a.jpg"
    with pytest.raises(DatasetError, match="missing"):
        load_items(path, check_media=True)
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    assert load_items(path, check_media=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"split": "validation"},
        {"expected_label": "genuine"},
        {"id": "Bad Id"},
        {"kind": "image"},  # no media path
        {"source_note": ""},
        {"unknown_field": 1},
    ],
)
def test_bad_rows_rejected(tmp_path: Path, bad: dict[str, Any]) -> None:
    with pytest.raises(DatasetError):
        load_items(write_jsonl(tmp_path / "a.jsonl", [row(**bad)]))


def test_language_must_match_script(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Devanagari"):
        load_items(write_jsonl(tmp_path / "a.jsonl", [row(language="hi")]))
    with pytest.raises(DatasetError, match="Devanagari"):
        load_items(write_jsonl(tmp_path / "b.jsonl", [row(text="यह हिंदी है")]))
    ok = row(text="yeh hinglish hai", language="hinglish")
    assert load_items(write_jsonl(tmp_path / "c.jsonl", [ok]))


def test_synthetic_flag_rules(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="synthetic"):
        load_items(write_jsonl(tmp_path / "a.jsonl", [row(source_note="SYNTHETIC: made up")]))
    fine = row(source_note="SYNTHETIC: made up", synthetic=True)
    assert load_items(write_jsonl(tmp_path / "b.jsonl", [fine]))[0].synthetic


def test_canary_only_in_injection_suite(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="canary"):
        load_items(write_jsonl(tmp_path / "a.jsonl", [row(canary="X-1")]))
    ok = row(suite="injection", canary="X-1")
    assert load_items(write_jsonl(tmp_path / "b.jsonl", [ok]))[0].canary == "X-1"


def test_placeholder_rows_rejected_under_real(tmp_path: Path) -> None:
    template = row(source_note="SYNTHETIC-PLACEHOLDER: replace", synthetic=True)
    real = tmp_path / "real"
    real.mkdir()
    write_jsonl(real / "a.jsonl", [template])
    with pytest.raises(DatasetError, match="placeholder"):
        load_with_folders([real])
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    write_jsonl(elsewhere / "a.jsonl", [template])
    items, folders = load_with_folders([elsewhere])
    assert folders["item-001"] == elsewhere
    assert items[0].synthetic


def test_bad_json_line_reports_position(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text("{not json}\n", "utf-8")
    with pytest.raises(DatasetError, match=r"a\.jsonl:1"):
        load_items(path)


def test_split_helpers_and_report() -> None:
    items = [
        Item.model_validate(row(id=f"item-{i:03d}", split="dev" if i % 2 else "test"))
        for i in range(4)
    ]
    assert [i.id for i in iter_split(items, "dev")] == ["item-001", "item-003"]
    assert len(list(iter_split(items, "all"))) == 4
    assert split_counts(items) == {"scam": {"test": 2, "dev": 2}}
    report = validate_dataset_report(items)
    assert report["counts"] == {"scam": 4}
    assert any("scam: 4 of 100" in w for w in report["warnings"])  # type: ignore[attr-defined]
