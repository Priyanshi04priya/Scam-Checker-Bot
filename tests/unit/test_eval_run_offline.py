"""The offline runner end to end on the committed seed data."""

import json
from pathlib import Path

import pytest
from eval import run
from eval.report import read_results_jsonl
from eval.schema import load_dir
from eval.tasks import Criterion, ItemResult, SuiteReport, evaluate_unverifiable

from scambot.core.config import AppConfig
from scambot.preprocess.hashing import text_sha256
from tests.media import jpeg_bytes

SEED = Path(run.__file__).parent / "data" / "seed"


@pytest.fixture(scope="module")
def seed_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("seed-run")
    assert run.main(["--mode", "offline", "--suite", "all", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def results(seed_run: Path) -> dict[str, ItemResult]:
    return {
        r.item_id: r
        for r in read_results_jsonl(seed_run / "results.jsonl")
        if "-c" not in r.item_id
    }


def of(results: dict[str, ItemResult], suite: str) -> list[ItemResult]:
    return [r for r in results.values() if r.suite == suite]


def test_every_seed_item_gets_a_reply(results: dict[str, ItemResult]) -> None:
    items = load_dir(SEED)
    assert {i.id for i in items} <= set(results)
    for result in results.values():
        assert result.error is None
        assert result.request_status == "replied"
        assert result.predicted_label is not None
        assert result.reply_text


def test_injection_items_get_normal_verdicts_without_canaries(
    results: dict[str, ItemResult],
) -> None:
    injection = of(results, "injection")
    assert len(injection) == 20
    for r in injection:
        assert r.canary
        assert r.canary not in r.reply_text
        assert r.predicted_label in {"no_issues", "likely_scam", "could_not_verify"}
    kept = [r for r in injection if r.category == "injection_scam"]
    assert kept
    assert all(r.predicted_label == "likely_scam" for r in kept)


def test_no_unverifiable_reply_calls_the_claim_genuine(
    results: dict[str, ItemResult], app_config: object
) -> None:
    assert isinstance(app_config, AppConfig)
    report = evaluate_unverifiable(
        of(results, "unverifiable"), app_config.thresholds.reply.genuine_terms
    )
    claim = next(c for c in report.criteria if "no reply claims genuineness" in c.name)
    assert claim.observed.startswith("0/")
    assert {c.status for c in report.criteria} == {"INCOMPLETE"}  # synthetic items never count


def test_factcheck_seed_items_match_their_synthetic_article(results: dict[str, ItemResult]) -> None:
    for r in of(results, "factcheck"):
        assert r.predicted_label == "false_misleading", r.item_id
        urls = [e.url for c in r.checks for e in c.evidence]
        assert r.expected_source in urls


def test_scam_metrics_are_probabilities_and_files_are_written(seed_run: Path) -> None:
    metrics = json.loads((seed_run / "metrics.json").read_text("utf-8"))
    scam = next(s for s in metrics["suites"] if s["suite"] == "scam")
    for criterion in scam["criteria"]:
        assert 0.0 <= float(criterion["observed"]) <= 1.0
        assert criterion["status"] == "INCOMPLETE"
    for name in (
        "results.jsonl", "results.csv", "metrics.json", "errors.csv", "audit.csv",
        "manual_review.csv", "report.md", "run_meta.json", "confusion_scam.csv",
    ):  # fmt: skip
        assert (seed_run / name).is_file(), name
    meta = json.loads((seed_run / "run_meta.json").read_text("utf-8"))
    assert meta["mode"] == "offline"
    assert meta["sources"]["offline"] > 0
    names = {Path(k).name for k in meta["config_sha256"]}
    assert names >= {"thresholds.yaml", "verdict_rules.yaml"}


def test_report_has_sections_and_incomplete(seed_run: Path) -> None:
    text = (seed_run / "report.md").read_text("utf-8")
    for heading in ("## Results", "## scam", "## factcheck", "## unverifiable", "## injection"):
        assert heading in text
    assert "INCOMPLETE" in text
    assert "not a measurement of real accuracy" in text
    assert "outage: skipped" in text


def test_cache_and_retention_pass(seed_run: Path) -> None:
    metrics = json.loads((seed_run / "metrics.json").read_text("utf-8"))
    by_suite = {s["suite"]: s for s in metrics["suites"]}
    assert {c["status"] for c in by_suite["cache"]["criteria"]} == {"PASS"}
    assert by_suite["retention"]["criteria"][0]["status"] == "PASS"


def test_outage_run_still_replies_and_names_the_missing_check(tmp_path: Path) -> None:
    out = tmp_path / "outage"
    argv = ["--suite", "outage", "--disable", "google_factcheck,anthropic", "--out", str(out)]
    assert run.main(argv) == 0
    metrics = json.loads((out / "metrics.json").read_text("utf-8"))
    outage = next(s for s in metrics["suites"] if s["suite"] == "outage")
    assert [c["status"] for c in outage["criteria"]] == [
        "INCOMPLETE",
        "PASS",
    ]  # anthropic: no check degraded
    rows = read_results_jsonl(out / "results.jsonl")
    degraded = [r for r in rows if r.unavailable]
    assert degraded
    assert all("fact-check search" in r.reply_text for r in degraded if r.reply_language == "en")
    assert any(r.reply_language == "en" for r in degraded)
    assert all(r.request_status == "replied" for r in rows)


def test_disable_without_outage_suite_applies_to_the_whole_run(tmp_path: Path) -> None:
    out = tmp_path / "disabled"
    argv = ["--suite", "factcheck", "--disable", "google_factcheck", "--out", str(out)]
    assert run.main(argv) == 0
    rows = read_results_jsonl(out / "results.jsonl")
    assert rows
    assert all(r.unavailable == ["fact_check_claim"] for r in rows)


def test_split_filter_and_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "dev"
    assert run.main(["--suite", "injection", "--split", "dev", "--out", str(out)]) == 0
    assert {r.split for r in read_results_jsonl(out / "results.jsonl")} == {"dev"}
    assert run.main(["--suite", "nonsense", "--out", str(out)]) == 2
    assert run.main(["--today", "yesterday", "--out", str(out)]) == 2
    assert run.main(["--mode", "live", "--replay-only", "--out", str(out)]) == 2
    assert "error:" in capsys.readouterr().err


def test_strict_fails_when_a_criterion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = SuiteReport(
        suite="x", n=1, criteria=[Criterion(name="c", target="t", observed="o", n=1, status="FAIL")]
    )

    async def fake_execute(
        plan: object, build: object
    ) -> tuple[list[ItemResult], list[SuiteReport], dict[str, object], dict[str, object]]:
        return [], [failing], {}, {"mode": "offline"}

    monkeypatch.setattr(run, "execute", fake_execute)
    assert run.main(["--suite", "scam", "--out", str(tmp_path / "a")]) == 0
    assert run.main(["--suite", "scam", "--strict", "--out", str(tmp_path / "b")]) == 1


def test_harness_errors_exit_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(plan: object, build: object) -> None:
        raise RuntimeError("broken")

    monkeypatch.setattr(run, "execute", boom)
    assert run.main(["--suite", "scam", "--out", str(tmp_path / "a")]) == 1


def test_outage_applicability() -> None:
    items = load_dir(SEED)
    assert {i.suite for i in run.outage_items("google_factcheck", items, 100)} == {"factcheck"}
    assert len(run.outage_items("anthropic", items, 7)) == 7
    assert run.outage_items("sightengine", items, 10) == []
    assert all("http" in i.text for i in run.outage_items("google_safe_browsing", items, 100))


def test_cache_variants() -> None:
    assert run.text_variant("Hello World") != "Hello World"
    assert text_sha256(run.text_variant("Hello World")) == text_sha256("Hello World")
    data = jpeg_bytes()
    assert run.recompressed(data) != data
