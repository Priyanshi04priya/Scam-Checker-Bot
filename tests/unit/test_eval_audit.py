"""The reply grounding audit."""

import datetime as dt
from pathlib import Path

from eval.audit import (
    audit_reply,
    audit_summary,
    extract_claims,
    normalise_digits,
    run_audit,
    sample_for_manual_review,
    template_numbers,
    write_audit_csv,
    write_manual_review,
)
from eval.tasks import ItemResult

from scambot.core.contracts import CheckerResult, Evidence
from scambot.core.locales import Locales


def result(reply: str, *, evidence: list[Evidence] | None = None, **over: object) -> ItemResult:
    checks = [
        CheckerResult(
            checker="fact_check_claim",
            status="ok",
            signal="debunked",
            confidence=0.9,
            evidence=evidence or [],
            duration_ms=1,
            details={"matches": [{"date": "2021-03-12"}]},
        )
    ]
    base = {
        "item_id": "i-1",
        "suite": "scam",
        "split": "dev",
        "kind": "text",
        "language": "en",
        "expected_label": "no_issues",
        "predicted_label": "false_misleading",
        "reply_text": reply,
        "checks": checks,
        "score": 82,
    }
    return ItemResult.model_validate(base | over)


GOOD = Evidence(
    summary="Fact-check: the claim is false (rated 100%)",
    url="https://factcheck.example.invalid/a/1?utm_source=x",
    date=dt.date(2021, 3, 12),
)


def test_grounded_reply_passes(locales: Locales) -> None:
    reply = (
        "Checked on 12 March 2021: see https://factcheck.example.invalid/a/1 "
        "(rating 100%). Score 82."
    )
    assert audit_reply(result(reply, evidence=[GOOD]), "", template_numbers(locales)) == []


def test_invented_url_date_and_number_fail(locales: Locales) -> None:
    reply = (
        "See https://invented.example.invalid/x, published 5 June 2019, and 4321 people shared it."
    )
    bad = audit_reply(result(reply, evidence=[GOOD]), "", template_numbers(locales))
    assert any(b.startswith("url:") for b in bad)
    assert any(b.startswith("date:2019-06") for b in bad)
    assert "number:4321" in bad


def test_numbers_from_the_users_own_text_are_allowed(locales: Locales) -> None:
    reply = "You asked about Rs 5,000 sent to 9000000000."
    user = "pay Rs 5000 to 9000000000"
    assert audit_reply(result(reply), user, set()) == []
    assert audit_reply(result(reply), "", set()) != []


def test_static_locale_numbers_are_ignored(locales: Locales) -> None:
    static = template_numbers(locales)
    assert "90" in static  # retention days in the privacy text
    assert audit_reply(result("Files over 20 MB are not checked."), "", static) == []


def test_hindi_digits_and_months() -> None:
    assert normalise_digits("४५६") == "456"
    claims = extract_claims("२५ मार्च २०२१ को ४५६ लोग")
    assert (2021, 3, 25) in claims["dates"]
    assert "456" in claims["numbers"]
    assert "2021" not in claims["numbers"]


def test_iso_and_year_only_dates() -> None:
    claims = extract_claims("on 2020-01-05 and in 2018, 7 times")
    assert claims["dates"] == {(2020, 1, 5), (2018, None, None)}
    assert claims["numbers"] == {"7"}


def test_run_audit_and_csv(locales: Locales, tmp_path: Path) -> None:
    rows = run_audit([result("fine, score 82"), result("invented 999", item_id="i-2")], {}, locales)
    assert [r["pass"] for r in rows] == [True, False]
    assert audit_summary(rows) == {"audited": 2, "failing": 1, "passing": 1}
    out = tmp_path / "audit.csv"
    write_audit_csv(rows, out)
    assert "number:999" in out.read_text("utf-8")


def many(n: int) -> list[ItemResult]:
    labels = ["likely_scam", "no_issues", "could_not_verify"]
    return [
        result(
            f"reply {i}",
            item_id=f"i-{i:03d}",
            predicted_label=labels[i % 3],
            language="hi" if i % 2 else "en",
        )
        for i in range(n)
    ]


def test_manual_sample_is_deterministic_stratified_and_sized(tmp_path: Path) -> None:
    rows = many(60)
    a = sample_for_manual_review(rows, 12, seed=1)
    b = sample_for_manual_review(rows, 12, seed=1)
    assert [r.item_id for r in a] == [r.item_id for r in b]
    assert len(a) == 12
    strata = {(r.predicted_label, r.language) for r in a}
    assert len(strata) == 6  # every (label, language) group is represented
    assert [r.item_id for r in sample_for_manual_review(rows, 12, seed=2)] != [r.item_id for r in a]
    assert len(sample_for_manual_review(rows[:5], 100)) == 5
    out = tmp_path / "m.csv"
    write_manual_review(a, out)
    assert out.read_text("utf-8").splitlines()[0].startswith("item_id,language,label")
