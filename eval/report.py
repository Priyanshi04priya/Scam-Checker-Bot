"""CSV and Markdown writers for a run, and the filled-in report template."""

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from eval.tasks import ItemResult

TEMPLATE: Final = Path(__file__).parent / "report_template.md"
RESULT_COLUMNS: Final = (
    "item_id", "suite", "split", "kind", "language", "category", "expected_label",
    "predicted_label", "band", "score", "duration_ms", "cached", "unavailable", "source", "error",
)  # fmt: skip
HONEST_LIMITS: Final = (
    "Seed data only, not a measurement of real accuracy. The items were written to check that "
    "the harness works; these numbers must not be quoted as results."
)


def write_results_jsonl(results: Sequence[ItemResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(r.model_dump_json() + "\n")


def read_results_jsonl(path: Path) -> list[ItemResult]:
    with path.open(encoding="utf-8") as handle:
        return [ItemResult.model_validate_json(line) for line in handle if line.strip()]


def write_results_csv(results: Sequence[ItemResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        for r in results:
            row = r.model_dump()
            row["unavailable"] = ";".join(r.unavailable)
            writer.writerow([row[c] for c in RESULT_COLUMNS])


def write_confusion_csv(matrix: Mapping[str, Mapping[str, int]], path: Path) -> None:
    labels = list(next(iter(matrix.values()), {}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected \\ predicted", *labels])
        for row_label, row in matrix.items():
            writer.writerow([row_label, *(row[c] for c in labels)])


def _ci(ci: Sequence[float] | None) -> str:
    return f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else "-"


def results_table(suites: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Metric | Target | Observed | n | 95% CI | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for suite in suites:
        for c in suite["criteria"]:
            lines.append(
                f"| {c['name']} | {c['target']} | {c['observed']} | {c['n']} | "
                f"{_ci(c.get('ci'))} | {c['status']} |"
            )
    return lines


def render_markdown(metrics: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    """The run report: one results table, then a section per suite."""
    suites: list[Mapping[str, Any]] = metrics["suites"]
    lines = ["# Evaluation report", ""]
    lines.append(
        f"Mode: {meta.get('mode', '?')} | run: {meta.get('timestamp', '?')} | "
        f"items: {meta.get('item_count', '?')}"
    )
    lines.append("")
    seed = bool(meta.get("seed_data"))
    shaky = [s["suite"] for s in suites if s.get("synthetic_share", 0) > 0.5]
    if seed or shaky:
        lines += [f"> {HONEST_LIMITS}", ""]
        if shaky:
            lines += [f"> Suites with mostly synthetic items: {', '.join(shaky)}.", ""]
    lines += ["## Results", "", *results_table(suites), ""]
    lines.append(
        "PASS needs the target met with the full item quota of real items; INCOMPLETE means the "
        "quota was not reached, so the number is not a pass or a fail."
    )
    lines.append("")
    for suite in suites:
        lines += [f"## {suite['suite']} (n={suite['n']})", ""]
        for note in suite.get("notes", []):
            lines.append(f"- {note}")
        bad = suite.get("misclassified", [])
        if bad:
            lines.append(f"- misclassified or failing: {', '.join(bad[:40])}")
        details = suite.get("details", {})
        for key in ("recall_by_language", "recall_by_category"):
            if key in details:
                shown = ", ".join(f"{k} {v:.2f}" for k, v in details[key].items())
                lines.append(f"- {key.replace('_', ' ')}: {shown}")
        lines.append("")
    audit = metrics.get("audit")
    if audit:
        lines += [
            "## Reply grounding audit",
            "",
            f"- audited {audit['audited']}, failing {audit['failing']} (see audit.csv)",
            "",
        ]
    skipped = metrics.get("skipped", [])
    if skipped:
        lines += ["## Skipped", "", *[f"- {s}" for s in skipped], ""]
    return "\n".join(lines).rstrip() + "\n"


def fill_template(template: str, values: Mapping[str, str]) -> str:
    """Replace ``{{name}}`` placeholders; unknown ones are left for a person to fill in."""
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
