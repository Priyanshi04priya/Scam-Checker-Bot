"""CSV and Markdown output of a run."""

import csv
from pathlib import Path

from eval import report
from eval.tasks import (
    Criterion,
    ItemResult,
    SuiteReport,
    evaluate_scam,
    grade,
)


def suite(*statuses: str, synthetic: float = 0.0, name: str = "scam") -> dict[str, object]:
    criteria = [
        Criterion(name=f"c{i}", target=">= 1", observed="1", n=3, status=s)  # type: ignore[arg-type]
        for i, s in enumerate(statuses)
    ]
    return SuiteReport(suite=name, n=3, criteria=criteria, synthetic_share=synthetic).model_dump(
        mode="json"
    )


def test_grade_needs_quota_and_target() -> None:
    assert grade(0.95, 0.9, 100, 100) == "PASS"
    assert grade(0.5, 0.9, 100, 100) == "FAIL"
    assert grade(0.99, 0.9, 99, 100) == "INCOMPLETE"
    assert grade(0, 0, 30, 30, higher=False) == "PASS"
    assert grade(1, 0, 30, 30, higher=False) == "FAIL"


def test_markdown_shows_each_status() -> None:
    text = report.render_markdown(
        {"suites": [suite("PASS", "FAIL", "INCOMPLETE")]}, {"mode": "live", "item_count": 3}
    )
    for word in ("PASS", "FAIL", "INCOMPLETE"):
        assert f"| {word} |" in text
    assert "not a measurement of real accuracy" not in text


def test_honest_limits_for_synthetic_majority_and_seed() -> None:
    text = report.render_markdown(
        {"suites": [suite("PASS", synthetic=0.8, name="injection")]}, {"mode": "live"}
    )
    assert "not a measurement of real accuracy" in text
    assert "injection" in text.split("mostly synthetic")[1]
    seed = report.render_markdown({"suites": [suite("PASS")]}, {"seed_data": True})
    assert "not a measurement of real accuracy" in seed


def test_misclassified_and_skipped_are_listed() -> None:
    s = SuiteReport(suite="scam", n=2, misclassified=["a-1", "a-2"], notes=["note here"])
    text = report.render_markdown(
        {"suites": [s.model_dump(mode="json")], "skipped": ["old_media: skipped"],
         "audit": {"audited": 5, "failing": 1}}, {},
    )  # fmt: skip
    assert "a-1, a-2" in text
    assert "note here" in text
    assert "old_media: skipped" in text
    assert "failing 1" in text


def result(item_id: str, expected: str, predicted: str, lang: str = "en") -> ItemResult:
    return ItemResult(
        item_id=item_id, suite="scam", split="dev", kind="text", language=lang,
        expected_label=expected, predicted_label=predicted, category="kyc",
        unavailable=["fact_check_claim"], duration_ms=12,
    )  # fmt: skip


def test_results_round_trip(tmp_path: Path) -> None:
    rows = [
        result("a-1", "likely_scam", "likely_scam"),
        result("a-2", "no_issues", "could_not_verify"),
    ]
    jsonl = tmp_path / "r.jsonl"
    report.write_results_jsonl(rows, jsonl)
    assert report.read_results_jsonl(jsonl) == rows
    csv_path = tmp_path / "r.csv"
    report.write_results_csv(rows, csv_path)
    with csv_path.open(encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert read[1]["predicted_label"] == "could_not_verify"
    assert read[0]["unavailable"] == "fact_check_claim"


def test_confusion_csv_orientation(tmp_path: Path) -> None:
    scam = evaluate_scam(
        [result("a-1", "likely_scam", "no_issues"), result("a-2", "no_issues", "no_issues")]
    )
    path = tmp_path / "c.csv"
    report.write_confusion_csv(scam.details["confusion"], path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    scam_row = next(r for r in rows if r[0] == "likely_scam")
    assert scam_row[header.index("no_issues")] == "1"
    assert scam.details["false_negatives"] == ["a-1"]


def test_fill_template_leaves_unknown_placeholders() -> None:
    template = Path(report.TEMPLATE).read_text("utf-8")
    filled = report.fill_template(template, {"project": "Demo"})
    assert filled.startswith("# Evaluation report: Demo")
    assert "{{summary}}" in filled
