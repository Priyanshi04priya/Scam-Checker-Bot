"""The evaluation folder and demo script carry no internal references."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
# Assembled from pieces so that this file does not itself contain the words it forbids.
_WORDS = [
    "local" + "_docs",
    r"\b(?:N?FR|AT|UC)-\d",
    r"\bH" + "LD\\b",
    r"\bL" + "LD\\b",
    "co-" + "authored",
]
FORBIDDEN = re.compile("|".join(_WORDS), re.IGNORECASE)
SUFFIXES = {".py", ".md", ".jsonl", ".json", ".csv", ".yaml", ".yml", ".txt"}


def files() -> list[Path]:
    found = [p for p in (ROOT / "eval").rglob("*") if p.is_file() and p.suffix in SUFFIXES]
    found += [ROOT / "DEMO.md", ROOT / "README.md"]
    return [p for p in found if "__pycache__" not in p.parts and "results" not in p.parts]


def test_no_internal_references_in_eval_demo_or_readme() -> None:
    assert files()
    for path in files():
        for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
            assert not FORBIDDEN.search(line), f"{path.relative_to(ROOT)}:{number}"


def test_gitignore_keeps_real_data_and_results_out() -> None:
    ignored = (ROOT / ".gitignore").read_text("utf-8")
    for pattern in ("eval/data/real/", "eval/results/", "*.replay.json"):
        assert pattern in ignored
