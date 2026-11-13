"""Threshold tuning: dev picks, test reports, config untouched."""

import hashlib
import re
from pathlib import Path

import pytest
from eval import tune
from eval.report import write_results_jsonl
from eval.tasks import ItemResult

from scambot.core.config import AppConfig
from scambot.core.contracts import CheckerResult
from scambot.core.settings import CONFIG_DIR


def scam_row(item_id: str, split: str, truth: str, confidence: float) -> ItemResult:
    check = CheckerResult(
        checker="analyze_scam_patterns",
        status="ok",
        signal="scam",
        confidence=confidence,
        duration_ms=1,
    )
    return ItemResult(
        item_id=item_id,
        suite="scam",
        split=split,
        kind="text",
        language="en",
        expected_label=truth,
        predicted_label="likely_scam",
        checks=[check],
        request_status="replied",
    )


def build_results() -> list[ItemResult]:
    """Dev: the 0.70 current cut-off misses a scam at 0.62; test would prefer a higher cut-off."""
    rows = [scam_row(f"d{i}", "dev", "likely_scam", 0.9) for i in range(4)]
    rows += [scam_row("d-low", "dev", "likely_scam", 0.62)]
    rows += [scam_row(f"dn{i}", "dev", "no_issues", 0.30) for i in range(5)]
    rows += [scam_row(f"t{i}", "test", "likely_scam", 0.9) for i in range(4)]
    rows += [scam_row("t-bad", "test", "no_issues", 0.65)]
    return rows


def test_parse_sweep() -> None:
    sweep = tune.parse_sweep("rule.scam=0.5:0.7:0.1")
    assert sweep.values == [0.5, 0.6, 0.7]
    assert tune.parse_sweep("x=1,2").values == [1.0, 2.0]
    for bad in ("x", "x=1:0:1", "x=0:1:0"):
        with pytest.raises(ValueError, match="bad"):
            tune.parse_sweep(bad)


async def test_best_value_is_chosen_on_dev_only(app_config: AppConfig) -> None:
    sweep = tune.parse_sweep("rule.scam=0.50:0.90:0.10")
    outcome = await tune.tune_sweep(sweep, build_results(), app_config)
    assert outcome.current == 0.70
    assert outcome.chosen == 0.60  # recall 5/5 on dev with precision 1.0; test would say 0.70
    assert outcome.dev[0.60]["recall"] == 1.0
    assert outcome.test_current["precision"] == 1.0
    assert outcome.test_chosen is not None
    assert outcome.test_chosen["precision"] < 1.0  # the held-out cost of that choice is shown
    assert "0.7 -> 0.6" in tune.suggestion(outcome)


async def test_no_feasible_value_keeps_current(app_config: AppConfig) -> None:
    rows = [scam_row(f"dn{i}", "dev", "no_issues", 0.95) for i in range(3)]
    outcome = await tune.tune_sweep(tune.parse_sweep("rule.scam=0.5:0.9:0.2"), rows, app_config)
    assert outcome.chosen is None
    assert "keep 0.7" in tune.suggestion(outcome)


async def test_internal_cutoffs_need_rerun(app_config: AppConfig) -> None:
    with pytest.raises(ValueError, match="--rerun"):
        await tune.tune_sweep(tune.parse_sweep("scam.signal_min=0.2:0.4:0.1"), [], app_config)
    with pytest.raises(ValueError, match="unknown"):
        await tune.tune_sweep(tune.parse_sweep("nonsense=1"), [], app_config)


async def test_rerun_hook_is_used_for_internal_cutoffs(app_config: AppConfig) -> None:
    calls: list[float] = []

    async def rerun(name: str, value: float) -> list[ItemResult]:
        calls.append(value)
        return build_results()

    sweep = tune.parse_sweep("scam.signal_min=0.3:0.5:0.1")
    outcome = await tune.tune_sweep(sweep, build_results(), app_config, rerun)
    assert calls == [0.3, 0.4, 0.5]
    assert outcome.current == app_config.thresholds.scam.signal_min


def test_override_rule_changes_a_copy(app_config: AppConfig) -> None:
    rules = app_config.verdict_rules
    changed = tune.override_rule(rules, "scam", 0.5)
    assert tune.current_value(changed, "scam") == 0.5
    assert tune.current_value(rules, "scam") == 0.70
    deep = tune.override_rule(rules, "synthetic", 0.5, "detect_voice_clone")
    assert tune.current_value(deep, "synthetic", "detect_voice_clone") == 0.5
    assert tune.current_value(deep, "synthetic", "detect_deepfake") == 0.85


def digest() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(CONFIG_DIR.glob("**/*"))
        if p.is_file()
    }


def test_cli_writes_reports_and_never_touches_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results.jsonl"
    write_results_jsonl(build_results(), results)
    before = digest()
    out = tmp_path / "tune"
    assert (
        tune.main(
            ["--results", str(results), "--sweep", "rule.scam=0.5:0.9:0.1", "--out", str(out)]
        )
        == 0
    )
    assert digest() == before
    printed = capsys.readouterr().out
    assert "suggested change" in printed
    assert "min_confidence" in printed
    assert (out / "sweep.csv").is_file()
    assert "held-out test" in (out / "tune_report.md").read_text("utf-8")


def test_refuses_without_dev_items(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_results_jsonl([r for r in build_results() if r.split == "test"], results)
    with pytest.raises(SystemExit) as caught:
        tune.main(["--results", str(results), "--out", str(tmp_path / "o")])
    assert caught.value.code == 2


def test_bad_sweep_exits_two(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_results_jsonl(build_results(), results)
    assert (
        tune.main(["--results", str(results), "--sweep", "bogus", "--out", str(tmp_path / "o")])
        == 2
    )


def test_the_module_has_no_way_to_write_config() -> None:
    source = Path(tune.__file__).read_text("utf-8")
    assert "CONFIG_DIR" not in source
    # Everything written goes under the --out folder.
    writes = re.findall(r"(\S+)\.(?:write_text|open)\(", source)
    assert writes
    assert all(
        w.endswith(("out", '"sweep.csv")', '"tune_report.md")')) or "out /" in w for w in writes
    )


def test_other_rules_use_their_own_objectives() -> None:
    row = ItemResult(
        item_id="f1", suite="factcheck", split="dev", kind="text", language="en",
        expected_label="false_misleading", predicted_label="false_misleading",
    )  # fmt: skip
    assert tune.setting_metrics([row], "debunked")["objective"] == 0
    assert tune.setting_metrics([row], "old_or_out_of_context")["feasible"] == 0.0
    assert tune.setting_metrics([row], "synthetic")["accuracy"] == 1.0
