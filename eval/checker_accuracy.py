"""Accuracy of the AI-image and synthetic-voice checks on labelled folders.

Usage::

    python -m eval.checker_accuracy deepfake --data DIR --out OUT
    python -m eval.checker_accuracy voice --manifest labels.csv --backend live --out OUT

``DIR`` holds ``real/`` and ``fake/`` subfolders; a manifest is a CSV with ``path,label`` rows
(label ``real`` or ``fake``, paths relative to the manifest). Each file goes through the same
checker class the bot uses. ``--backend fake`` swaps the vendor for a scripted stand-in, for
dry runs and tests; ``--backend live`` needs the usual keys or model.
"""

import argparse
import asyncio
import csv
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from eval.metrics import equal_error_rate, percentile, precision_recall_f1, sweep
from eval.report import write_json
from eval.voice_compare import AUDIO_SUFFIXES
from scambot.core.config import AppConfig, load_config
from scambot.core.contracts import CheckerResult
from scambot.core.media_store import AUDIO_FILE, ORIGINAL, LocalMediaStore
from scambot.core.settings import get_settings
from scambot.mcp_server.clients.sightengine import SynthScores
from scambot.preprocess.stt import to_wav

IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".webp"})
CUTOFFS: Final = [round(0.1 * i, 1) for i in range(1, 10)]
CAVEAT: Final = (
    "These checks are shown to users as a low-confidence signal only: a high score never "
    "decides a verdict alone, and a low score does not show the content is genuine."
)
FAKE_HIGH: Final = 0.9
FAKE_LOW: Final = 0.1


@dataclass(frozen=True)
class Entry:
    path: Path
    label: int  # 1 = fake


@dataclass
class Row:
    path: str
    label: int
    status: str
    signal: str
    score: float | None
    latency_ms: int


def collect(kind: str, data: Path | None, manifest: Path | None) -> list[Entry]:
    suffixes = IMAGE_SUFFIXES if kind == "deepfake" else AUDIO_SUFFIXES
    entries: list[Entry] = []
    if manifest is not None:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                word = row["label"].strip().lower()
                if word not in {"real", "fake"}:
                    msg = f"{manifest.name}: label must be real or fake, got {row['label']!r}"
                    raise ValueError(msg)
                label = 1 if word == "fake" else 0
                entries.append(Entry(manifest.parent / row["path"], label))
    elif data is not None:
        for name, label in (("real", 0), ("fake", 1)):
            folder = data / name
            if folder.is_dir():
                entries += [
                    Entry(p, label)
                    for p in sorted(folder.rglob("*"))
                    if p.is_file() and p.suffix.lower() in suffixes
                ]
    return entries


class _ScriptedAnalyzer:
    """Fake image vendor: scores whatever label the runner says the current file has."""

    def __init__(self) -> None:
        self.label = 0

    async def scores(self, image: bytes, filename: str) -> SynthScores:
        high = self.label == 1
        return SynthScores(FAKE_HIGH if high else FAKE_LOW, FAKE_HIGH if high else FAKE_LOW)


class _ScriptedDetector:
    name = "fake:voice"

    def __init__(self) -> None:
        self.label = 0

    def score(self, audio: Any) -> float:
        return FAKE_HIGH if self.label == 1 else FAKE_LOW

    def warm_up(self) -> None:
        return None


async def _check_deepfake(
    checker: Any, store: LocalMediaStore, entry: Entry, key: str
) -> CheckerResult:
    async def chunks() -> Any:
        yield entry.path.read_bytes()

    await store.put(key, ORIGINAL, chunks(), max_bytes=50_000_000)
    result: CheckerResult = await checker.check(key, [ORIGINAL])
    return result


async def _check_voice(
    checker: Any, store: LocalMediaStore, entry: Entry, key: str, config: AppConfig
) -> CheckerResult:
    async def one_byte() -> AsyncIterator[bytes]:
        yield b"x"

    await store.put(key, "placeholder", one_byte(), max_bytes=1)
    target = store.path(key, "placeholder").with_name(AUDIO_FILE)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        to_wav(ffmpeg, entry.path, target, config.thresholds.media.stt)
    else:
        shutil.copyfile(entry.path, target)
    result: CheckerResult = await checker.check(key, AUDIO_FILE)
    return result


@dataclass
class Runner:
    """A checker with the store and settings it needs, and the fake's label hook if any."""

    kind: str
    checker: Any
    store: LocalMediaStore
    config: AppConfig
    script: Any | None = None


async def run_checker(runner: Runner, entries: Sequence[Entry]) -> list[Row]:
    rows: list[Row] = []
    for number, entry in enumerate(entries):
        if runner.script is not None:
            runner.script.label = entry.label
        key = f"{number:032x}"
        started = time.perf_counter()
        try:
            if runner.kind == "deepfake":
                result = await _check_deepfake(runner.checker, runner.store, entry, key)
            else:
                result = await _check_voice(runner.checker, runner.store, entry, key, runner.config)
            status, signal = result.status, result.signal
            raw = result.details.get("score")
            score = float(raw) if isinstance(raw, int | float) else None
        except Exception as exc:  # a file the checker cannot read is a result, not a crash
            sys.stderr.write(f"warning: {entry.path.name}: {type(exc).__name__}\n")
            status, signal, score = "error", "none", None
        runner.store.delete(key)
        elapsed = round((time.perf_counter() - started) * 1000)
        rows.append(Row(entry.path.name, entry.label, status, signal, score, elapsed))
    return rows


def summarise(kind: str, rows: Sequence[Row], signal_min: float) -> dict[str, Any]:
    ok = [r for r in rows if r.status == "ok" and r.score is not None]
    scores = [r.score for r in ok if r.score is not None]
    truths = [r.label == 1 for r in ok]
    tp = sum(1 for s, t in zip(scores, truths, strict=True) if s >= signal_min and t)
    fp = sum(1 for s, t in zip(scores, truths, strict=True) if s >= signal_min and not t)
    fn = sum(1 for s, t in zip(scores, truths, strict=True) if s < signal_min and t)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    table = sweep(scores, truths, CUTOFFS)
    best = max(table, key=lambda row: row["f1"]) if table else None
    eer, eer_cut = equal_error_rate([int(t) for t in truths], scores)
    latencies = [r.latency_ms / 1000 for r in rows]
    return {
        "kind": kind,
        "files": len(rows),
        "ok": len(ok),
        "inconclusive": sum(1 for r in rows if r.status == "inconclusive"),
        "error": sum(1 for r in rows if r.status == "error"),
        "coverage": len(ok) / len(rows) if rows else 0.0,
        "signal_min": signal_min,
        "accuracy": (
            tp + sum(1 for s, t in zip(scores, truths, strict=True) if s < signal_min and not t)
        )
        / len(ok)
        if ok
        else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sweep": table,
        "best_f1_cutoff_informational": best["cutoff"] if best else None,
        "eer": eer if kind == "voice" else None,
        "eer_threshold": eer_cut if kind == "voice" else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
    }


def write_outputs(rows: Sequence[Row], metrics: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label", "status", "signal", "score", "latency_ms"])
        for r in rows:
            score = "" if r.score is None else f"{r.score:.4f}"
            label = "fake" if r.label else "real"
            writer.writerow([r.path, label, r.status, r.signal, score, r.latency_ms])
    write_json(metrics, out / "metrics.json")
    lines = [
        f"# {metrics['kind']} accuracy",
        "",
        f"> {CAVEAT}",
        "",
        f"- files: {metrics['files']} (ok {metrics['ok']}, inconclusive {metrics['inconclusive']}, "
        f"error {metrics['error']}); coverage {metrics['coverage']:.1%}",
        f"- at the configured cut-off {metrics['signal_min']}: accuracy {metrics['accuracy']:.3f}, "
        f"precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, "
        f"F1 {metrics['f1']:.3f}",
        f"- best-F1 cut-off (informational, do not tune on this alone): "
        f"{metrics['best_f1_cutoff_informational']}",
        f"- latency p50 {metrics['latency_p50_s']:.3f} s, p95 {metrics['latency_p95_s']:.3f} s",
    ]
    if metrics["eer"] is not None:
        lines.append(
            f"- equal error rate {metrics['eer']:.3f} at cut-off {metrics['eer_threshold']:.3f}"
        )
    lines += ["", "| cut-off | precision | recall | F1 |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {r['cutoff']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        for r in metrics["sweep"]
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_checker(kind: str, config: AppConfig, store: LocalMediaStore) -> tuple[Any, Any]:
    from tests.fakes import speech_vad  # noqa: PLC0415 - fake backend only

    from scambot.mcp_server.checkers.deepfake import DeepfakeChecker  # noqa: PLC0415
    from scambot.mcp_server.checkers.voice_clone import VoiceCloneChecker  # noqa: PLC0415
    from scambot.mcp_server.media import MediaFiles  # noqa: PLC0415

    media = MediaFiles(store)
    if kind == "deepfake":
        analyzer = _ScriptedAnalyzer()
        return DeepfakeChecker(analyzer, media, config.thresholds.deepfake), analyzer
    detector = _ScriptedDetector()
    return VoiceCloneChecker(detector, media, config.thresholds.voice, speech_vad()), detector


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.checker_accuracy", description=__doc__)
    p.add_argument("kind", choices=["deepfake", "voice"])
    p.add_argument("--data", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--backend", choices=["live", "fake"], default="live")
    p.add_argument("--out", type=Path, required=True)
    return p


async def run(args: argparse.Namespace) -> int:
    if (args.data is None) == (args.manifest is None):
        sys.stderr.write("error: give exactly one of --data and --manifest\n")
        return 2
    entries = collect(args.kind, args.data, args.manifest)
    if not entries:
        sys.stderr.write("error: no files found\n")
        return 2
    config = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "media"
        store = LocalMediaStore(root)
        closeables: list[Any] = []
        script: Any | None = None
        if args.backend == "fake":
            checker, script = _fake_checker(args.kind, config, store)
        else:
            from scambot.mcp_server.server import create_deps  # noqa: PLC0415

            settings = get_settings().model_copy(update={"media_dir": root})
            deps, closeables = create_deps(settings, config)
            checker = deps.deepfake if args.kind == "deepfake" else deps.voice
        try:
            rows = await run_checker(Runner(args.kind, checker, store, config, script), entries)
        finally:
            for item in closeables:
                await item.aclose()
    limit = config.thresholds.deepfake if args.kind == "deepfake" else config.thresholds.voice
    metrics = summarise(args.kind, rows, limit.signal_min)
    write_outputs(rows, metrics, args.out)
    sys.stdout.write(f"wrote {args.out}/report.md ({len(rows)} files)\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from scambot.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("WARNING")
    return asyncio.run(run(_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
