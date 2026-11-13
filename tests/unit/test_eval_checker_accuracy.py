"""Deepfake and voice accuracy runs on tiny generated folders, with the scripted fake vendor."""

import json
from pathlib import Path

import pytest
from eval import checker_accuracy

from tests.fakes import noisy_jpeg, write_tone


def image_folder(root: Path, real: int = 3, fake: int = 3, small: int = 0) -> Path:
    for name, count in (("real", real), ("fake", fake)):
        (root / name).mkdir(parents=True)
        for i in range(count):
            (root / name / f"{name}{i}.jpg").write_bytes(noisy_jpeg(300, hash(name) % 1000 + i))
    for i in range(small):  # too small to judge: the checker is inconclusive
        (root / "real" / f"tiny{i}.jpg").write_bytes(noisy_jpeg(32, i))
    return root


def test_deepfake_fake_backend_scores_and_excludes_inconclusive(tmp_path: Path) -> None:
    data = image_folder(tmp_path / "data", small=2)
    out = tmp_path / "out"
    code = checker_accuracy.main(
        ["deepfake", "--data", str(data), "--backend", "fake", "--out", str(out)]
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["files"] == 8
    assert metrics["ok"] == 6
    assert metrics["inconclusive"] == 2
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == metrics["recall"] == 1.0
    assert len(metrics["sweep"]) == 9
    scores = (out / "scores.csv").read_text("utf-8")
    assert "fake0.jpg,fake,ok,likely_synthetic,0.9000" in scores
    report = (out / "report.md").read_text("utf-8")
    assert "low-confidence signal only" in report


def test_manifest_mode(tmp_path: Path) -> None:
    data = image_folder(tmp_path / "data", 2, 2)
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "path,label\ndata/real/real0.jpg,real\ndata/fake/fake0.jpg,fake\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    assert data.is_dir()
    assert (
        checker_accuracy.main(
            ["deepfake", "--manifest", str(manifest), "--backend", "fake", "--out", str(out)]
        )
        == 0
    )
    assert json.loads((out / "metrics.json").read_text("utf-8"))["files"] == 2


def test_voice_fake_backend_reports_eer(tmp_path: Path) -> None:
    for name in ("real", "fake"):
        (tmp_path / "v" / name).mkdir(parents=True)
        for i in range(3):
            write_tone(tmp_path / "v" / name / f"{name}{i}.wav", 4.0)
    out = tmp_path / "out"
    code = checker_accuracy.main(
        ["voice", "--data", str(tmp_path / "v"), "--backend", "fake", "--out", str(out)]
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    assert metrics["ok"] == 6
    assert metrics["eer"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_argument_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert checker_accuracy.main(["deepfake", "--backend", "fake", "--out", str(tmp_path)]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    code = checker_accuracy.main(
        ["deepfake", "--data", str(empty), "--backend", "fake", "--out", str(tmp_path / "o")]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_summarise_handles_no_ok_rows() -> None:
    rows = [checker_accuracy.Row("a.jpg", 1, "error", "none", None, 5)]
    metrics = checker_accuracy.summarise("deepfake", rows, 0.5)
    assert metrics["ok"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["best_f1_cutoff_informational"] is None or isinstance(
        metrics["best_f1_cutoff_informational"], float
    )
