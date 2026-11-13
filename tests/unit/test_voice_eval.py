"""Tests for the detector comparison script, with synthetic audio and fake detectors."""

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from eval.voice_compare import (
    RESULT_COLUMNS,
    SCORE_COLUMNS,
    Clip,
    compute_metrics,
    main,
    run_comparison,
)

from scambot.core.config import AppConfig, VoiceThresholds
from scambot.mcp_server.clients.voice_spoof import VoiceSpoofDetector, VoiceSpoofError
from scambot.preprocess.stt import SAMPLE_RATE, Audio
from tests.fakes import speech_vad, write_tone


class EnergyDetector:
    """Scores loud clips as fake (or the reverse when ``inverted``)."""

    def __init__(self, name: str = "energy", *, inverted: bool = False) -> None:
        self.name = name
        self._inverted = inverted

    def score(self, audio: Audio) -> float:
        value = 0.9 if float(np.abs(audio).mean()) > 0.05 else 0.1
        return 1.0 - value if self._inverted else value

    def warm_up(self) -> None:
        return None


def clip(label: int, loud: bool, seconds: float = 5.0) -> Clip:
    rng = np.random.default_rng(label + int(loud))
    count = round(seconds * SAMPLE_RATE)
    audio = (rng.normal(0, 0.3 if loud else 0.01, count)).astype(np.float32)
    return Clip(Path(f"{'fake' if label else 'real'}_{loud}.wav"), label, audio)


def test_perfect_and_reversed_detectors(app_config: AppConfig) -> None:
    clips = [clip(1, True), clip(0, False), clip(1, True), clip(0, False)]
    cfg = app_config.thresholds.voice

    good, bad = run_comparison(
        clips,
        {"good": EnergyDetector(), "bad": EnergyDetector(inverted=True)},
        speech_vad(),
        cfg,
    )

    assert (good.accuracy_at_half, good.eer) == (1.0, 0.0)
    assert (bad.accuracy_at_half, bad.eer) == (0.0, 1.0)
    assert good.n_real == good.n_fake == 2
    assert good.mean_latency_s >= 0


def test_short_clips_are_skipped(app_config: AppConfig) -> None:
    clips = [clip(1, True, 1.0), clip(0, False), clip(1, True)]

    (result,) = run_comparison(
        clips, {"d": EnergyDetector()}, speech_vad(), app_config.thresholds.voice
    )

    assert result.skipped == 1
    assert result.n_real + result.n_fake == 2


def test_metrics_hand_cases() -> None:
    tied = compute_metrics([0, 1], [0.5, 0.5])
    one_class = compute_metrics([1, 1], [0.2, 0.9])
    empty = compute_metrics([], [])
    overlap = compute_metrics([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.6, 0.4, 0.8, 0.9])

    assert tied.eer == 0.5
    assert math.isnan(one_class.eer)
    assert one_class.accuracy_at_half == 0.5
    assert math.isnan(empty.accuracy_at_half)
    assert overlap.eer == pytest.approx(1 / 3)
    assert overlap.accuracy_at_best == pytest.approx(5 / 6)


def make_dataset(root: Path) -> None:
    for name in ("real", "fake"):
        (root / name).mkdir(parents=True)
    write_tone(root / "real" / "a.wav", 5)
    write_tone(root / "fake" / "b.wav", 5)
    write_tone(root / "fake" / "short.wav", 1)


def fake_build(name: str, spec: str, cfg: VoiceThresholds) -> VoiceSpoofDetector:
    if spec == "hosted":
        raise VoiceSpoofError("not_implemented")
    return EnergyDetector(name)


def test_main_writes_both_csv_files(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    make_dataset(tmp_path / "data")
    monkeypatch.setattr("eval.voice_compare.silero_vad", lambda _ms: speech_vad())
    out, scores = tmp_path / "r.csv", tmp_path / "s.csv"

    code = main(
        [
            "--data", str(tmp_path / "data"),
            "--candidate", "a=x",
            "--candidate", "h=hosted",
            "--out", str(out),
            "--scores-out", str(scores),
        ],
        build=fake_build,
    )  # fmt: skip

    assert code == 0
    rows = list(csv.reader(out.read_text().splitlines()))
    assert tuple(rows[0]) == RESULT_COLUMNS
    assert len(rows) == 2  # the hosted stub is skipped
    assert rows[1][0] == "a"
    assert rows[1][3] == "1"  # the 1 s clip
    score_rows = list(csv.reader(scores.read_text().splitlines()))
    assert tuple(score_rows[0]) == SCORE_COLUMNS
    assert len(score_rows) == 3


def test_main_needs_data_and_candidates(tmp_path: Path) -> None:
    assert main(["--candidate", "a=x"]) == 2
    assert main(["--data", str(tmp_path)]) == 2
