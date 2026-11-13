"""Run the evaluation: ``python -m eval.run --mode offline|live ...`` (see eval/README.md)."""

import argparse
import asyncio
import datetime as dt
import hashlib
import io
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from PIL import Image

from eval import audit, report
from eval.backends import Stack, build_live_stack, build_offline_stack, load_articles, missing_keys
from eval.replay import SEED_REPLAY, SEED_TODAY, VENDORS, ReplayStore
from eval.schema import DatasetError, Item, iter_split, load_with_folders, validate_dataset_report
from eval.tasks import (
    Criterion,
    ItemResult,
    SuiteReport,
    evaluate_cache,
    evaluate_factcheck,
    evaluate_injection,
    evaluate_latency,
    evaluate_old_media,
    evaluate_outage,
    evaluate_retention,
    evaluate_scam,
    evaluate_unverifiable,
    grade,
    latency_summary,
    run_item,
    run_items,
)
from scambot.core.config import load_config
from scambot.core.locales import load_locales
from scambot.core.logging import configure_logging
from scambot.core.settings import CONFIG_DIR, get_settings

SEED_DIR: Final = Path(__file__).parent / "data" / "seed"
RESULTS_DIR: Final = Path(__file__).parent / "results"
ALL_SUITES: Final = (
    "scam", "factcheck", "old_media", "unverifiable", "injection", "outage", "cache", "retention",
)  # fmt: skip
ITEM_SUITES: Final = ("scam", "factcheck", "old_media", "unverifiable", "injection", "genuine")
CACHE_SAMPLE: Final = 10

StackBuilder = Callable[..., Awaitable[Stack]]


def recompressed(data: bytes) -> bytes:
    """The same picture saved again as a lower-quality JPEG (what a re-forward does)."""
    buffer = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


def text_variant(text: str) -> str:
    """A trivial edit the cache must see through: other case, doubled spaces."""
    return "  " + text.upper().replace(" ", "  ") + " "


def outage_items(vendor: str, items: Sequence[Item], limit: int) -> list[Item]:
    """The first ``limit`` items (by id) that the vendor's check applies to."""

    def applies(item: Item) -> bool:
        if vendor == "anthropic":
            return item.kind == "text" and item.suite in {"scam", "injection", "factcheck"}
        if vendor == "google_factcheck":
            return item.suite == "factcheck"
        if vendor == "google_safe_browsing":
            return item.suite == "scam" and "http" in item.text
        if vendor in {"google_vision", "sightengine"}:
            return item.kind == "image"
        return item.kind in {"audio", "voice"}

    return sorted((i for i in items if applies(i)), key=lambda i: i.id)[:limit]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.run", description="Run the evaluation.")
    p.add_argument("--mode", choices=["offline", "live"], default="offline")
    p.add_argument("--data", type=Path, nargs="+", default=None)
    p.add_argument("--suite", default="all")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--replay-file", type=Path, default=None)
    p.add_argument("--replay-only", action="store_true")
    p.add_argument("--disable", default="", help="comma separated vendors to switch off")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--today", default=None, help="YYYY-MM-DD used as the received date")
    p.add_argument("--database-url", default=None)
    p.add_argument("--audit-sample", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outage-sample", type=int, default=10)
    p.add_argument("--strict", action="store_true", help="exit 1 when any criterion FAILs")
    return p


def _hash_files(paths: Sequence[Path]) -> dict[str, str]:
    """SHA-256 of each file, keyed by its path relative to the working directory when possible."""
    out: dict[str, str] = {}
    for p in paths:
        if not p.is_file():
            continue
        try:
            key = p.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            key = p.resolve().as_posix()
        out[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _redact(name: str, value: object) -> str:
    """An argument for run_meta.json; a database URL can carry a password, so it is masked."""
    return "***" if name == "database_url" and value else str(value)


def _sources(results: Sequence[ItemResult]) -> dict[str, int]:
    return dict(Counter(r.source for r in results))


def _vendor_calls(results: Sequence[ItemResult]) -> dict[str, int]:
    calls: Counter[str] = Counter()
    for r in results:
        for c in r.checks:
            reported = c.details.get("vendor_calls")
            if isinstance(reported, dict):
                calls.update({str(k): int(v) for k, v in reported.items() if isinstance(v, int)})
    return dict(calls)


async def _default_builder(**kw: Any) -> Stack:
    if kw["mode"] == "offline":
        return await build_offline_stack(
            replay=kw["replay"], disabled=kw["disabled"], articles=kw["articles"]
        )
    return await build_live_stack(
        get_settings(),
        load_config(),
        replay=kw["replay"],
        replay_only=kw["replay_only"],
        disabled=kw["disabled"],
        database_url=kw["database_url"],
    )


def _usable(item: Item, folders: dict[str, Path]) -> bool:
    if item.kind == "text":
        return True
    return bool(item.media_path and (folders[item.id] / item.media_path).is_file())


def _read(path: Path) -> bytes:
    return path.read_bytes()


@dataclass
class Plan:
    """What a run will do, worked out before any async work starts."""

    args: argparse.Namespace
    suites: set[str]
    items: list[Item]
    folders: dict[str, Path]
    runnable: list[Item]
    skipped: list[str]
    paths: list[Path]
    today: str
    disabled: tuple[str, ...]
    articles: list[Any]
    replay: ReplayStore

    @property
    def stack_args(self) -> dict[str, Any]:
        outage_on = "outage" in self.suites
        return {
            "mode": self.args.mode,
            "replay": self.replay,
            "disabled": () if outage_on else self.disabled,
            "articles": self.articles,
            "replay_only": self.args.replay_only,
            "database_url": self.args.database_url,
        }


def prepare(args: argparse.Namespace, suites: set[str]) -> Plan:
    paths: list[Path] = list(args.data or [SEED_DIR])
    items, folders = load_with_folders(paths)
    if args.today:
        today = args.today
    elif args.mode == "offline":
        today = SEED_TODAY
    else:
        today = dt.datetime.now(dt.UTC).date().isoformat()
    replay_path = args.replay_file
    if replay_path is None:
        replay_path = (
            SEED_REPLAY if args.mode == "offline" else args.out.with_suffix(".replay.json")
        )
    chosen = _select(items, suites, args.split)
    runnable = [i for i in chosen if _usable(i, folders)]
    skipped: list[str] = []
    if len(runnable) != len(chosen):
        skipped.append(f"{len(chosen) - len(runnable)} media items skipped (no media file found)")
    if args.mode == "live" and args.database_url is None:
        if any(i.suite == "factcheck" for i in runnable):
            skipped.append("factcheck: skipped (no index); pass --database-url to enable it")
        runnable = [i for i in runnable if i.suite != "factcheck"]
    articles = [
        a for p in paths if p.is_dir() for a in load_articles(p / "synthetic_articles.jsonl")
    ]
    return Plan(
        args, suites, items, folders, runnable, skipped, paths, today,
        tuple(v for v in args.disable.split(",") if v), articles,
        ReplayStore(replay_path, autosave=args.mode == "live"),
    )  # fmt: skip


async def _cache_pairs(build: StackBuilder, plan: Plan) -> list[tuple[ItemResult, ItemResult]]:
    candidates = [
        i for i in sorted(plan.items, key=lambda i: i.id) if i.suite in {"scam", "old_media"}
    ]
    chosen = [i for i in candidates if _usable(i, plan.folders)][:CACHE_SAMPLE]
    stack = await build(**plan.stack_args)
    pairs: list[tuple[ItemResult, ItemResult]] = []
    try:
        for item in chosen:
            folder = plan.folders[item.id]
            data = _read(folder / item.media_path) if item.media_path else None
            first = await run_item(stack, item, today=plan.today, folder=folder, suffix="-c1")
            second = await run_item(
                stack,
                item,
                today=plan.today,
                suffix="-c2",
                text=text_variant(item.text) if item.kind == "text" else None,
                media_bytes=recompressed(data) if data and item.kind == "image" else data,
            )
            pairs.append((first, second))
    finally:
        await stack.aclose()
    return pairs


async def _outage(build: StackBuilder, plan: Plan) -> dict[str, list[ItemResult]]:
    out: dict[str, list[ItemResult]] = {}
    for vendor in plan.disabled:
        picked = outage_items(vendor, plan.items, plan.args.outage_sample)
        picked = [i for i in picked if _usable(i, plan.folders)]
        results: list[ItemResult] = []
        if picked:
            stack = await build(**{**plan.stack_args, "disabled": (vendor,)})
            try:
                results = await run_items(
                    stack, picked, today=plan.today, folders=plan.folders, concurrency=2
                )
            finally:
                await stack.aclose()
        out[vendor] = results
    return out


def _write(
    out: Path,
    results: Sequence[ItemResult],
    suites: list[SuiteReport],
    extra: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"suites": [s.model_dump(mode="json") for s in suites], **extra}
    report.write_results_jsonl(results, out / "results.jsonl")
    report.write_results_csv(results, out / "results.csv")
    report.write_json(metrics, out / "metrics.json")
    report.write_json(meta, out / "run_meta.json")
    for s in suites:
        if "confusion" in s.details:
            report.write_confusion_csv(s.details["confusion"], out / f"confusion_{s.suite}.csv")
    bad = {i for s in suites for i in s.misclassified}
    report.write_results_csv([r for r in results if r.item_id in bad], out / "errors.csv")
    (out / "report.md").write_text(report.render_markdown(metrics, meta), encoding="utf-8")


def _select(items: Sequence[Item], suites: set[str], split: str) -> list[Item]:
    wanted = set(suites)
    if "old_media" in wanted:
        wanted.add("genuine")
    return [i for i in iter_split(items, split) if i.suite in wanted and i.suite in ITEM_SUITES]


def _item_reports(plan: Plan, by_suite: dict[str, list[ItemResult]]) -> list[SuiteReport]:
    suites = plan.suites
    genuine_terms = load_config().thresholds.reply.genuine_terms
    reports: list[SuiteReport] = []
    if "scam" in suites and by_suite.get("scam"):
        reports.append(evaluate_scam(by_suite["scam"]))
    if "factcheck" in suites and by_suite.get("factcheck"):
        reports.append(evaluate_factcheck(by_suite["factcheck"]))
    if "old_media" in suites:
        if by_suite.get("old_media"):
            reports.append(evaluate_old_media(by_suite["old_media"], by_suite.get("genuine", [])))
        else:
            plan.skipped.append("old_media: skipped (no media in this run)")
    if "unverifiable" in suites and by_suite.get("unverifiable"):
        reports.append(evaluate_unverifiable(by_suite["unverifiable"], genuine_terms))
    if "injection" in suites and by_suite.get("injection"):
        reports.append(evaluate_injection(by_suite["injection"]))
    return reports


def _grounding(
    results: Sequence[ItemResult], plan: Plan
) -> tuple[dict[str, int], SuiteReport | None]:
    rows = audit.run_audit(results, {i.id: i.text for i in plan.items}, load_locales())
    out: Path = plan.args.out
    out.mkdir(parents=True, exist_ok=True)
    audit.write_audit_csv(rows, out / "audit.csv")
    sample = audit.sample_for_manual_review(results, plan.args.audit_sample, plan.args.seed)
    audit.write_manual_review(sample, out / "manual_review.csv")
    stats = audit.audit_summary(rows)
    if not stats["audited"]:
        return stats, None
    criterion = Criterion(
        name="reply grounding audit (automated)",
        target="0 unsupported",
        observed=f"{stats['failing']} of {stats['audited']} replies",
        n=stats["audited"],
        status=grade(stats["failing"], 0, stats["audited"], 100, higher=False),
    )
    return stats, SuiteReport(suite="grounding", n=stats["audited"], criteria=[criterion])


def _meta(plan: Plan, everything: Sequence[ItemResult]) -> dict[str, Any]:
    args = plan.args
    cost = sum((Decimal(r.cost_usd) for r in everything), Decimal(0))
    files = [p for d in plan.paths if d.is_dir() for p in sorted(d.glob("*.jsonl"))]
    files += [p for p in plan.paths if p.is_file()]
    replay = plan.replay
    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "mode": args.mode,
        "today": plan.today,
        "seed_data": all(SEED_DIR in (p.resolve(), *p.resolve().parents) for p in plan.paths),
        "item_count": len(everything),
        "dataset": validate_dataset_report(plan.items),
        "dataset_sha256": _hash_files(files),
        "config_sha256": _hash_files(sorted(CONFIG_DIR.glob("*.yaml"))),
        "sources": _sources(everything),
        "replay": {"hits": replay.hits, "misses": replay.misses, "recorded": replay.recorded},
        "cost_usd": str(cost),
        "vendor_calls": _vendor_calls(everything),
        "disabled": list(plan.disabled),
        "args": {k: _redact(k, v) for k, v in vars(args).items()},
    }


async def execute(
    plan: Plan, build: StackBuilder
) -> tuple[list[ItemResult], list[SuiteReport], dict[str, Any], dict[str, Any]]:
    stack = await build(**plan.stack_args)
    try:
        results = await run_items(
            stack, plan.runnable, today=plan.today, folders=plan.folders,
            concurrency=plan.args.concurrency,
        )  # fmt: skip
        retention = evaluate_retention(stack.media_root, len(results))
    finally:
        await stack.aclose()
    by_suite: dict[str, list[ItemResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)
    reports = _item_reports(plan, by_suite)
    extra_results: list[ItemResult] = []
    if "outage" in plan.suites:
        if plan.disabled:
            per_vendor = await _outage(build, plan)
            reports.append(evaluate_outage(per_vendor, load_locales()))
            extra_results += [r for rows in per_vendor.values() for r in rows]
        else:
            plan.skipped.append("outage: skipped (pass --disable VENDOR[,VENDOR] to run it)")
    if "cache" in plan.suites:
        pairs = await _cache_pairs(build, plan)
        if pairs:
            reports.append(evaluate_cache(pairs))
            extra_results += [r for pair in pairs for r in pair]
        else:
            plan.skipped.append("cache: skipped (no usable items)")
    if "retention" in plan.suites:
        reports.append(retention)
    latency = latency_summary(results)
    if latency:
        reports.append(evaluate_latency(latency))
    audit_stats, grounding = _grounding(results, plan)
    if grounding is not None:
        reports.append(grounding)
    everything = [*results, *extra_results]
    extra = {"audit": audit_stats, "skipped": plan.skipped, "latency": latency}
    return everything, reports, extra, _meta(plan, everything)


def _check_args(args: argparse.Namespace, suites: set[str], vendors: list[str]) -> str | None:
    """A message for the first problem with the arguments, or None."""
    if not suites <= set(ALL_SUITES) or not set(vendors) <= set(VENDORS) or args.concurrency < 1:
        return "unknown suite or vendor, or bad concurrency"
    if args.today:
        try:
            dt.date.fromisoformat(args.today)
        except ValueError:
            return "--today must be YYYY-MM-DD"
    if (
        args.mode == "live"
        and args.replay_only
        and not (args.replay_file and args.replay_file.is_file())
    ):
        return "--replay-only needs an existing --replay-file"
    return None


def main(argv: Sequence[str] | None = None, *, builder: StackBuilder | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging("WARNING")
    suites = set(ALL_SUITES) if args.suite == "all" else {s for s in args.suite.split(",") if s}
    vendors = [v for v in args.disable.split(",") if v]
    problem = _check_args(args, suites, vendors)
    if problem is None and args.mode == "live" and builder is None:
        missing = missing_keys(get_settings(), vendors)
        if missing:
            problem = f"live mode needs {', '.join(missing)} (or --disable the vendor)"
    if problem is not None:
        sys.stderr.write(f"error: {problem}\n")
        return 2
    if args.out is None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        args.out = RESULTS_DIR / f"run-{stamp}"
    try:
        plan = prepare(args, suites)
        results, reports, extra, meta = asyncio.run(execute(plan, builder or _default_builder))
    except DatasetError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"harness error: {type(exc).__name__}: {exc}\n")
        return 1
    _write(args.out, results, reports, extra, meta)
    failed = [c.name for s in reports for c in s.criteria if c.status == "FAIL"]
    sys.stdout.write(f"wrote {args.out}/report.md ({len(results)} results, {len(failed)} FAIL)\n")
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
