"""Compare synthetic-voice detectors on labelled audio.

This is run by hand with real model weights and data; it is not part of the automated tests.
A comparison of two or three open anti-spoofing models has NOT been run in this repository, so
the default model id in the settings is provisional until this script has been run.

Usage::

    python -m eval.voice_compare --data DIR --candidate a=MODEL_ID --candidate b=MODEL_ID \\
        --out results.csv --scores-out scores.csv

``DIR`` holds ``real/`` and ``fake/`` subfolders (any format ffmpeg reads), or pass
``--manifest labels.csv`` with ``path,label`` rows where label is ``real`` or ``fake``. The
candidate value ``reality_defender`` selects the hosted stub, which is skipped with a warning.

Suggested data: the ASVspoof 2019 LA evaluation subset, In-the-Wild, and clips of your own,
such as Hindi and English text-to-speech against real Telegram voice notes (OGG/Opus), because
compression is the realistic weakness.

Each clip goes through the same steps as the live check: decode to 16 kHz mono, find speech,
pick the densest window, skip clips with too little speech, then score. Use the printed numbers
to set ``voice.signal_min`` and the verdict rule.
"""

import argparse
import csv
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scambot.core.config import SttThresholds, VoiceThresholds, load_config
from scambot.mcp_server.checkers.voice_clone import select_segment
from scambot.mcp_server.clients.voice_spoof import (
    HfAudioSpoofDetector,
    VoiceSpoofDetector,
    VoiceSpoofError,
)
from scambot.preprocess.base import MediaToolError
from scambot.preprocess.stt import Audio, Vad, read_wav, silero_vad, to_wav

RESULT_COLUMNS = (
    "candidate",
    "n_real",
    "n_fake",
    "skipped",
    "accuracy_at_0.5",
    "best_threshold",
    "accuracy_at_best",
    "eer",
    "eer_threshold",
    "mean_latency_s",
    "p95_latency_s",
)
SCORE_COLUMNS = ("candidate", "file", "label", "score")
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac"})


@dataclass(frozen=True)
class Clip:
    path: Path
    label: int  # 1 = fake, 0 = real
    audio: Audio


@dataclass
class Metrics:
    candidate: str = ""
    n_real: int = 0
    n_fake: int = 0
    skipped: int = 0
    accuracy_at_half: float = float("nan")
    best_threshold: float = float("nan")
    accuracy_at_best: float = float("nan")
    eer: float = float("nan")
    eer_threshold: float = float("nan")
    mean_latency_s: float = float("nan")
    p95_latency_s: float = float("nan")
    scores: list[tuple[str, int, float]] = field(default_factory=list)

    def row(self) -> list[object]:
        return [
            self.candidate,
            self.n_real,
            self.n_fake,
            self.skipped,
            self.accuracy_at_half,
            self.best_threshold,
            self.accuracy_at_best,
            self.eer,
            self.eer_threshold,
            self.mean_latency_s,
            self.p95_latency_s,
        ]


def compute_metrics(labels: Sequence[int], scores: Sequence[float]) -> Metrics:
    """Accuracy and equal error rate from labels (1 = fake) and scores; pure numpy."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    metrics = Metrics(n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()))
    if len(y) == 0:
        return metrics
    metrics.accuracy_at_half = float(((s >= 0.5) == (y == 1)).mean())
    best_acc, best_t = -1.0, float("nan")
    gap, eer, eer_t = float("inf"), float("nan"), float("nan")
    for threshold in np.unique(s):
        predicted = s >= threshold
        accuracy = float((predicted == (y == 1)).mean())
        if accuracy > best_acc:
            best_acc, best_t = accuracy, float(threshold)
        if metrics.n_real and metrics.n_fake:
            far = float(predicted[y == 0].mean())
            frr = float((~predicted)[y == 1].mean())
            if abs(far - frr) < gap:
                gap, eer, eer_t = abs(far - frr), (far + frr) / 2, float(threshold)
    metrics.best_threshold, metrics.accuracy_at_best = best_t, best_acc
    metrics.eer, metrics.eer_threshold = eer, eer_t
    return metrics


def run_comparison(
    clips: Sequence[Clip],
    candidates: Mapping[str, VoiceSpoofDetector],
    vad: Vad,
    cfg: VoiceThresholds,
) -> list[Metrics]:
    """Score every usable clip with every candidate, as the live check would."""
    prepared: list[tuple[Clip, Audio]] = []
    skipped = 0
    for clip in clips:
        spans = vad(clip.audio)
        segment, speech_s = select_segment(clip.audio, spans, cfg) if spans else (clip.audio, 0.0)
        if speech_s < cfg.min_speech_s:
            skipped += 1
        else:
            prepared.append((clip, segment))
    results: list[Metrics] = []
    for name, detector in candidates.items():
        labels: list[int] = []
        scores: list[float] = []
        latencies: list[float] = []
        rows: list[tuple[str, int, float]] = []
        for clip, segment in prepared:
            started = time.perf_counter()
            value = float(detector.score(segment))
            latencies.append(time.perf_counter() - started)
            labels.append(clip.label)
            scores.append(value)
            rows.append((str(clip.path), clip.label, value))
        metrics = compute_metrics(labels, scores)
        metrics.candidate, metrics.skipped, metrics.scores = name, skipped, rows
        if latencies:
            metrics.mean_latency_s = statistics.fmean(latencies)
            metrics.p95_latency_s = float(np.percentile(latencies, 95))
        results.append(metrics)
    return results


def _decode(path: Path, ffmpeg: str, scratch: Path, stt: SttThresholds) -> Audio:
    target = scratch / "clip.wav"
    to_wav(ffmpeg, path, target, stt)
    return read_wav(target)


def load_clips(data: Path | None, manifest: Path | None, cfg_stt: SttThresholds) -> list[Clip]:
    entries: list[tuple[Path, int]] = []
    if manifest is not None:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = 1 if row["label"].strip().lower() == "fake" else 0
                entries.append((manifest.parent / row["path"], label))
    elif data is not None:
        for name, label in (("real", 0), ("fake", 1)):
            folder = data / name
            entries += [
                (p, label)
                for p in sorted(folder.rglob("*"))
                if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
            ]
    ffmpeg = shutil.which("ffmpeg")
    clips: list[Clip] = []
    with tempfile.TemporaryDirectory() as tmp:
        for path, label in entries:
            try:
                if ffmpeg is None:
                    audio = read_wav(path)
                else:
                    audio = _decode(path, ffmpeg, Path(tmp), cfg_stt)
            except MediaToolError:
                sys.stderr.write(f"warning: could not read {path}\n")
                continue
            clips.append(Clip(path, label, audio))
    return clips


Builder = Callable[[str, str, VoiceThresholds], VoiceSpoofDetector]


def build_detector(name: str, spec: str, cfg: VoiceThresholds) -> VoiceSpoofDetector:
    """A detector for ``spec``: a Hugging Face model id, or ``reality_defender``."""
    if spec == "reality_defender":
        raise VoiceSpoofError("not_implemented")
    return HfAudioSpoofDetector(
        spec, revision=None, spoof_labels=cfg.spoof_labels, window_s=cfg.window_s
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.voice_compare", description="Compare synthetic-voice detectors."
    )
    parser.add_argument("--data", type=Path, help="folder with real/ and fake/ subfolders")
    parser.add_argument("--manifest", type=Path, help="CSV with path,label (real or fake)")
    parser.add_argument(
        "--candidate", action="append", default=[], metavar="NAME=MODEL_ID", help="repeatable"
    )
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    parser.add_argument("--scores-out", type=Path)
    parser.add_argument("--window-s", type=int)
    parser.add_argument("--min-speech-s", type=float)
    return parser


def main(argv: Sequence[str] | None = None, *, build: Builder = build_detector) -> int:
    args = _parser().parse_args(argv)
    if (args.data is None) == (args.manifest is None):
        sys.stderr.write("error: give exactly one of --data and --manifest\n")
        return 2
    if not args.candidate:
        sys.stderr.write("error: give at least one --candidate NAME=MODEL_ID\n")
        return 2
    config = load_config()
    cfg = config.thresholds.voice
    overrides = {
        key: value
        for key, value in (("window_s", args.window_s), ("min_speech_s", args.min_speech_s))
        if value is not None
    }
    cfg = cfg.model_copy(update=overrides)
    detectors: dict[str, VoiceSpoofDetector] = {}
    for item in args.candidate:
        name, _, spec = item.partition("=")
        if not spec:
            sys.stderr.write(f"error: bad candidate {item!r}\n")
            return 2
        try:
            detector = build(name, spec, cfg)
            detector.warm_up()
            detectors[name] = detector
        except (VoiceSpoofError, ValueError) as exc:
            sys.stderr.write(f"warning: skipping {name}: {exc}\n")
    clips = load_clips(args.data, args.manifest, config.thresholds.media.stt)
    metrics = run_comparison(clips, detectors, silero_vad(cfg.min_silence_ms), cfg)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        writer.writerows(m.row() for m in metrics)
    if args.scores_out is not None:
        with args.scores_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SCORE_COLUMNS)
            for m in metrics:
                writer.writerows(
                    (m.candidate, file, "fake" if label else "real", score)
                    for file, label, score in m.scores
                )
    sys.stdout.write(
        f"Results written to {args.out}. Set voice.signal_min and the verdict rule "
        "from the measured numbers.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
