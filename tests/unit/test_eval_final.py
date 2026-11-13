"""Evaluation edge cases: outage not exercised, confirmed-source grading, stable files."""

import json
from pathlib import Path

from eval import run, tune
from eval.replay import ReplayStore
from eval.tasks import ItemResult, evaluate_outage, evaluate_unverifiable

from scambot.core.locales import load_locales


def row(item_id: str = "u-1", **kw: object) -> ItemResult:
    base = {
        "item_id": item_id,
        "suite": "unverifiable",
        "split": "dev",
        "kind": "text",
        "language": "en",
        "expected_label": "could_not_verify",
        "predicted_label": "could_not_verify",
        "request_status": "replied",
        "reply_text": "Could not verify this.",
    }
    return ItemResult.model_validate(base | kw)


def test_outage_with_nothing_degraded_is_not_exercised() -> None:
    report = evaluate_outage({"anthropic": [row(), row("u-2")]}, load_locales())
    assert [c.status for c in report.criteria] == ["INCOMPLETE"]
    assert "not exercised" in report.criteria[0].observed


def test_confirmed_source_counts_as_genuine_unless_expected() -> None:
    terms = ["genuine"]
    bad = row(source_confirmed=True)
    ok = row("u-2", source_confirmed=True, expected_source="https://x.example.invalid/a")
    report = evaluate_unverifiable([bad, ok, row("u-3")], terms)
    assert report.misclassified == ["u-1"]
    assert report.criteria[0].observed.startswith("1/")


def test_replay_file_bytes_do_not_depend_on_insertion_order(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    one, two = ReplayStore(a), ReplayStore(b)
    one.put("k1", {"z": 1, "a": 2})
    one.put("k2", {"b": 1})
    two.put("k2", {"b": 1})
    two.put("k1", {"a": 2, "z": 1})
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(a.read_text("utf-8"))["entries"]["k1"] == {"a": 2, "z": 1}


def test_dataset_hashes_are_keyed_by_path(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    first, second = tmp_path / "x" / "items.jsonl", tmp_path / "y" / "items.jsonl"
    first.write_text("1", encoding="utf-8")
    second.write_text("2", encoding="utf-8")
    assert len(run._hash_files([first, second])) == 2


def test_tune_has_no_live_mode() -> None:
    parser = tune._parser()
    assert parser.parse_args(["--results", "r.jsonl"]).replay_file is None
    assert not any("--mode" in a.option_strings for a in parser._actions)
