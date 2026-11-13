"""Summarise a Locust stats CSV (and optionally the database) against the latency targets.

``python -m eval.loadtest.summarise --csv eval/results/load_stats.csv``
"""

import argparse
import asyncio
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from eval.metrics import percentile

# Request name -> (label, p95 target in seconds).
TARGETS: Final[dict[str, tuple[str, float]]] = {
    "e2e_text": ("text", 15.0),
    "e2e_image": ("image", 30.0),
    "e2e_audio": ("audio", 60.0),
    "e2e_video": ("video", 60.0),
    "webhook_text": ("ack text", 2.0),
    "webhook_image": ("ack image", 2.0),
    "webhook_audio": ("ack audio", 2.0),
    "webhook_video": ("ack video", 2.0),
}
HARD_CAP_S: Final = 90.0
DB_TARGETS_S: Final = {"text": 15.0, "image": 30.0, "audio": 60.0, "voice": 60.0, "video": 60.0}


@dataclass(frozen=True)
class Line:
    name: str
    count: int
    failures: int
    p50_s: float
    p95_s: float
    max_s: float
    target_s: float

    @property
    def status(self) -> str:
        if self.count == 0:
            return "INCOMPLETE"
        ok = self.p95_s <= self.target_s and self.max_s <= HARD_CAP_S * 1.0 and self.failures == 0
        return "PASS" if ok else "FAIL"


def parse_stats(path: Path) -> list[Line]:
    """Rows of a Locust ``*_stats.csv`` for the request names that have a target."""
    lines: list[Line] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Name", "")
            if name not in TARGETS:
                continue
            label, target = TARGETS[name]
            lines.append(
                Line(
                    name=label,
                    count=int(row["Request Count"]),
                    failures=int(row["Failure Count"]),
                    p50_s=float(row["50%"]) / 1000,
                    p95_s=float(row["95%"]) / 1000,
                    max_s=float(row["Max Response Time"]) / 1000,
                    target_s=target,
                )
            )
    return lines


def lines_from_durations(durations_ms: Mapping[str, Sequence[int]]) -> list[Line]:
    """Latency lines from verdict durations by input type (the real-instance run)."""
    out: list[Line] = []
    for kind, values in sorted(durations_ms.items()):
        seconds = [v / 1000 for v in values]
        out.append(
            Line(
                name=f"verdict {kind}",
                count=len(seconds),
                failures=0,
                p50_s=percentile(seconds, 50),
                p95_s=percentile(seconds, 95),
                max_s=max(seconds, default=0.0),
                target_s=DB_TARGETS_S.get(kind, 60.0),
            )
        )
    return out


def render(lines: Sequence[Line], first: int | None = None) -> str:
    rows = [
        "# Load test summary",
        "",
        f"p95 targets per input type; hard cap {HARD_CAP_S:g} s for any single request.",
        "" if first is None else f"Only the first {first} completed requests were counted.",
        "",
        "| Input | Requests | Failures | p50 s | p95 s | max s | Target p95 s | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows += [
        f"| {ln.name} | {ln.count} | {ln.failures} | {ln.p50_s:.2f} | {ln.p95_s:.2f} | "
        f"{ln.max_s:.2f} | {ln.target_s:g} | {ln.status} |"
        for ln in lines
    ]
    return "\n".join(rows) + "\n"


async def read_durations(database_url: str, since: str) -> dict[str, list[int]]:
    """Verdict durations by input type from the requests table (needs a database)."""
    import datetime as dt  # noqa: PLC0415

    from pydantic import SecretStr  # noqa: PLC0415
    from sqlalchemy import select  # noqa: PLC0415

    from scambot.core.db import create_engine  # noqa: PLC0415
    from scambot.core.models import Request  # noqa: PLC0415

    engine = create_engine(SecretStr(database_url))
    out: dict[str, list[int]] = {}
    try:
        async with engine.connect() as connection:
            cursor = await connection.execute(
                select(Request.kind, Request.duration_ms).where(
                    Request.created_at >= dt.datetime.fromisoformat(since),
                    Request.duration_ms.is_not(None),
                )
            )
            for kind, duration in cursor:
                out.setdefault(kind, []).append(int(duration or 0))
    finally:
        await engine.dispose()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.loadtest.summarise")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--database-url", default=None)
    p.add_argument("--since", default=None, help="ISO time; with --database-url")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if not args.csv.is_file():
        sys.stderr.write(f"error: {args.csv} not found\n")
        return 2
    lines = parse_stats(args.csv)
    if args.database_url:
        if not args.since:
            sys.stderr.write("error: --database-url needs --since\n")
            return 2
        lines += lines_from_durations(asyncio.run(read_durations(args.database_url, args.since)))
    text = render(lines)
    out = args.out or args.csv.with_name("load_summary.md")
    out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
