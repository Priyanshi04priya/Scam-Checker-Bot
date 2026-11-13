"""Threshold tuning on the dev split, reported on the held-out split.

``python -m eval.tune --results results.jsonl --out DIR`` re-applies the verdict rules to the
stored checker results for each candidate cut-off, so it costs nothing. The setting is chosen
on ``dev`` items only; the chosen and the current settings are then scored once on ``test``.
This module reads the config files and never writes them: the suggested change is printed for a
person to apply.

Cut-offs inside a checker (``scam.signal_min``, ``provenance.old_media_days``,
``deepfake.signal_min``, ``voice.signal_min``) change what the checker returns, so they need
``--rerun``, which executes the stack again from recorded traffic (``--replay-file``, default
the seed file; never a live vendor call) with the override held in memory only.
"""

import argparse
import asyncio
import csv
import math
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from eval.metrics import precision_recall_f1
from eval.report import read_results_jsonl
from eval.tasks import ItemResult, _source_matches
from scambot.agent.verdict import decide
from scambot.core.config import AppConfig, Clause, Condition, Rule, VerdictRulesConfig, load_config

INTERNAL: Final = {
    "scam.signal_min": ("scam", "signal_min"),
    "provenance.old_media_days": ("provenance", "old_media_days"),
    "deepfake.signal_min": ("deepfake", "signal_min"),
    "voice.signal_min": ("voice", "signal_min"),
}
DEFAULT_SWEEP: Final = "rule.scam=0.50:0.95:0.05"
PRECISION_FLOOR: Final = 0.90
OLD_MEDIA_PRECISION_FLOOR: Final = 0.85

Rerun = Callable[[str, float], Awaitable[list[ItemResult]]]


@dataclass(frozen=True)
class Sweep:
    name: str
    values: list[float]


def parse_sweep(spec: str) -> Sweep:
    """``name=lo:hi:step`` (inclusive) or ``name=a,b,c``."""
    name, _, rest = spec.partition("=")
    if not name or not rest:
        msg = f"bad --sweep {spec!r}"
        raise ValueError(msg)
    if ":" in rest:
        lo, hi, step = (float(x) for x in rest.split(":"))
        if step <= 0 or hi < lo:
            msg = f"bad range in {spec!r}"
            raise ValueError(msg)
        count = math.floor((hi - lo) / step + 1e-6)  # never past hi
        values = [round(lo + i * step, 6) for i in range(count + 1)]
    else:
        values = [float(x) for x in rest.split(",")]
    return Sweep(name, values)


def _clauses(rule: Rule, checker: str | None) -> list[Clause]:
    return [c for c in rule.when.any_of or [] if checker in (None, c.checker)]


def current_value(rules: VerdictRulesConfig, rule_id: str, checker: str | None = None) -> float:
    rule = next(r for r in rules.rules if r.id == rule_id)
    clauses = _clauses(rule, checker)
    if not clauses:
        msg = f"rule {rule_id} has no matching clause"
        raise ValueError(msg)
    return clauses[0].min_confidence


def override_rule(
    rules: VerdictRulesConfig, rule_id: str, value: float, checker: str | None = None
) -> VerdictRulesConfig:
    """A copy of ``rules`` where the rule's clause cut-offs are ``value`` (memory only)."""
    changed: list[Rule] = []
    for rule in rules.rules:
        if rule.id != rule_id or not rule.when.any_of:
            changed.append(rule)
            continue
        clauses = [
            c.model_copy(update={"min_confidence": value}) if checker in (None, c.checker) else c
            for c in rule.when.any_of
        ]
        changed.append(rule.model_copy(update={"when": Condition(any_of=clauses)}))
    return rules.model_copy(update={"rules": changed})


def relabel(rows: Sequence[ItemResult], rules: VerdictRulesConfig) -> list[ItemResult]:
    """Re-apply the verdict rules to the stored checker results."""
    out: list[ItemResult] = []
    for row in rows:
        if not row.checks or row.kind not in {"text", "image", "video", "audio", "voice"}:
            out.append(row)
            continue
        verdict = decide(row.checks, row.kind, rules)  # type: ignore[arg-type]
        out.append(
            row.model_copy(
                update={
                    "predicted_label": verdict.label,
                    "band": verdict.band,
                    "score": verdict.score,
                    "unavailable": list(verdict.unavailable),
                    "source_confirmed": verdict.source_confirmed,
                }
            )
        )
    return out


def _pred(r: ItemResult) -> str:
    return r.predicted_label or "error"


def setting_metrics(rows: Sequence[ItemResult], rule_id: str) -> dict[str, float]:
    """The numbers one setting is judged on, for the rule being swept."""
    if rule_id == "scam":
        scam = [r for r in rows if r.suite == "scam"]
        tp = sum(1 for r in scam if r.expected_label == "likely_scam" and _pred(r) == "likely_scam")
        fp = sum(1 for r in scam if r.expected_label != "likely_scam" and _pred(r) == "likely_scam")
        fn = sum(1 for r in scam if r.expected_label == "likely_scam" and _pred(r) != "likely_scam")
        p, rec, f1 = precision_recall_f1(tp, fp, fn)
        feasible = tp + fp > 0 and p >= PRECISION_FLOOR
        return {
            "precision": p,
            "recall": rec,
            "f1": f1,
            "objective": rec,
            "feasible": float(feasible),
        }
    if rule_id == "debunked":
        fact = [r for r in rows if r.suite == "factcheck"]
        matches = sum(1 for r in fact if _pred(r) == "false_misleading" and _source_matches(r))
        genuine = sum(1 for r in rows if r.suite == "unverifiable" and _pred(r) == "no_issues")
        wrong = sum(1 for r in rows if r.suite == "unverifiable" and _pred(r) == "false_misleading")
        return {
            "matches": matches,
            "unverifiable_genuine": genuine,
            "objective": matches,
            "feasible": float(wrong == 0),
        }
    if rule_id == "old_or_out_of_context":
        old = [r for r in rows if r.suite == "old_media"]
        control = [r for r in rows if r.suite == "genuine"]
        hits = sum(1 for r in old if _pred(r) == "old_or_out_of_context")
        flagged = sum(1 for r in control if _pred(r) == "old_or_out_of_context")
        prec = hits / (hits + flagged) if hits + flagged else 0.0
        rec = hits / len(old) if old else 0.0
        feasible = hits + flagged > 0 and prec >= OLD_MEDIA_PRECISION_FLOOR
        return {"precision": prec, "recall": rec, "objective": rec, "feasible": float(feasible)}
    ok = sum(1 for r in rows if _pred(r) == r.expected_label)
    acc = ok / len(rows) if rows else 0.0
    return {"accuracy": acc, "objective": acc, "feasible": 1.0}


def choose(table: dict[float, dict[str, float]], current: float) -> float | None:
    """Best feasible value (highest objective; ties go to the value closest to ``current``)."""
    feasible = {v: m for v, m in table.items() if m["feasible"]}
    if not feasible:
        return None
    best = max(m["objective"] for m in feasible.values())
    tied = [v for v, m in feasible.items() if m["objective"] == best]
    return min(tied, key=lambda v: (abs(v - current), v))


@dataclass
class TuneOutcome:
    name: str
    current: float
    chosen: float | None
    dev: dict[float, dict[str, float]]
    test_current: dict[str, float]
    test_chosen: dict[str, float] | None


async def tune_sweep(
    sweep: Sweep,
    results: Sequence[ItemResult],
    config: AppConfig,
    rerun: Rerun | None = None,
) -> TuneOutcome:
    """Pick the best value on dev, then score chosen and current on test, once."""
    if sweep.name.startswith("rule."):
        _, rule_id, *rest = sweep.name.split(".")
        checker = rest[0] if rest else None
        current = current_value(config.verdict_rules, rule_id, checker)

        async def rows_for(value: float) -> list[ItemResult]:
            return relabel(results, override_rule(config.verdict_rules, rule_id, value, checker))

        current_rows = relabel(results, config.verdict_rules)
    elif sweep.name in INTERNAL:
        if rerun is None:
            msg = f"{sweep.name} changes what a checker returns: pass --rerun with --data"
            raise ValueError(msg)
        section, field_name = INTERNAL[sweep.name]
        current = float(getattr(getattr(config.thresholds, section), field_name))
        rule_id = {"scam": "scam", "provenance": "old_or_out_of_context"}.get(section, "synthetic")

        async def rows_for(value: float) -> list[ItemResult]:
            return await rerun(sweep.name, value)

        current_rows = list(results)
    else:
        msg = f"unknown sweep {sweep.name!r}"
        raise ValueError(msg)
    table: dict[float, dict[str, float]] = {}
    cache: dict[float, list[ItemResult]] = {}
    for value in sweep.values:
        rows = await rows_for(value)
        cache[value] = rows
        table[value] = setting_metrics([r for r in rows if r.split == "dev"], rule_id)
    chosen = choose(table, current)
    test_current = setting_metrics([r for r in current_rows if r.split == "test"], rule_id)
    test_chosen = None
    if chosen is not None:
        chosen_rows = cache[chosen]
        test_chosen = setting_metrics([r for r in chosen_rows if r.split == "test"], rule_id)
    return TuneOutcome(sweep.name, current, chosen, table, test_current, test_chosen)


def suggestion(outcome: TuneOutcome) -> str:
    if outcome.chosen is None:
        return f"# {outcome.name}: no setting met the constraint on dev; keep {outcome.current}"
    if outcome.chosen == outcome.current:
        return f"# {outcome.name}: the current value {outcome.current} is already the best on dev"
    if outcome.name.startswith("rule."):
        parts = outcome.name.split(".")
        return (
            "# suggested change (apply by hand) in config/verdict_rules.yaml\n"
            f"# rule {parts[1]}: min_confidence: {outcome.current} -> {outcome.chosen}"
        )
    section, field_name = INTERNAL[outcome.name]
    return (
        "# suggested change (apply by hand) in config/thresholds.yaml\n"
        f"# {section}.{field_name}: {outcome.current} -> {outcome.chosen}"
    )


def write_outputs(outcomes: Sequence[TuneOutcome], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fields = ["sweep", "value", "split", "chosen", "current", "metrics"]
    lines = ["# Threshold tuning", "", "Chosen on the dev split only; test is scored once.", ""]
    with (out / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for o in outcomes:
            for value, metrics in sorted(o.dev.items()):
                shown = {k: round(v, 4) for k, v in metrics.items()}
                writer.writerow(
                    [o.name, value, "dev", value == o.chosen, value == o.current, shown]
                )
            writer.writerow([o.name, o.current, "test", False, True, o.test_current])
            if o.chosen is not None and o.test_chosen is not None:
                writer.writerow([o.name, o.chosen, "test", True, False, o.test_chosen])
            lines += [
                f"## {o.name}",
                "",
                f"- current: {o.current}, chosen on dev: {o.chosen}",
                f"- held-out test with current: {_fmt(o.test_current)}",
                f"- held-out test with chosen: {_fmt(o.test_chosen) if o.test_chosen else 'n/a'}",
                "",
                "```",
                suggestion(o),
                "```",
                "",
            ]
    text = "\n".join(lines)
    (out / "tune_report.md").write_text(text, encoding="utf-8")
    return text


def _fmt(metrics: dict[str, float] | None) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in (metrics or {}).items())


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.tune", description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--sweep", action="append", default=[])
    p.add_argument("--out", type=Path, default=Path("eval/results/tune"))
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--data", type=Path, nargs="+", default=None)
    p.add_argument(
        "--replay-file",
        type=Path,
        default=None,
        help="recorded vendor traffic for --rerun (replay only: tuning never calls a vendor)",
    )
    return p


def _make_rerun(args: argparse.Namespace, config: AppConfig) -> Rerun:
    from eval import run  # noqa: PLC0415 - only needed with --rerun
    from eval.backends import build_offline_stack, load_articles, with_thresholds  # noqa: PLC0415
    from eval.replay import SEED_REPLAY, SEED_TODAY, ReplayStore  # noqa: PLC0415
    from eval.schema import load_with_folders  # noqa: PLC0415
    from eval.tasks import run_items  # noqa: PLC0415

    paths = args.data or [run.SEED_DIR]
    items, folders = load_with_folders(paths)
    items = [i for i in items if run._usable(i, folders)]
    articles = [
        a for p in paths if p.is_dir() for a in load_articles(p / "synthetic_articles.jsonl")
    ]

    async def rerun(name: str, value: float) -> list[ItemResult]:
        section, field_name = INTERNAL[name]
        part = getattr(config.thresholds, section).model_copy(update={field_name: value})
        changed = with_thresholds(config, **{section: part})
        stack = await build_offline_stack(
            replay=ReplayStore(args.replay_file or SEED_REPLAY, autosave=False),
            config=changed,
            articles=articles,
        )
        try:
            return await run_items(stack, items, today=SEED_TODAY, folders=folders)
        finally:
            await stack.aclose()

    return rerun


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from scambot.core.logging import configure_logging  # noqa: PLC0415

    configure_logging("WARNING")
    results = read_results_jsonl(args.results)
    if not any(r.split == "dev" for r in results):
        sys.stderr.write("error: the results contain no dev items, nothing to tune on\n")
        raise SystemExit(2)
    config = load_config()
    try:
        sweeps = [parse_sweep(s) for s in (args.sweep or [DEFAULT_SWEEP])]
        rerun = _make_rerun(args, config) if args.rerun else None

        async def go() -> list[TuneOutcome]:
            return [await tune_sweep(s, results, config, rerun) for s in sweeps]

        outcomes = asyncio.run(go())
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    text = write_outputs(outcomes, args.out)
    sys.stdout.write(text + "\n")
    sys.stdout.write("\n".join(suggestion(o) for o in outcomes) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
