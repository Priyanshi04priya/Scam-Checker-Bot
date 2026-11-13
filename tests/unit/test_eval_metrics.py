"""Pure metric functions of the evaluation framework."""

import statistics

import pytest
from eval.metrics import (
    confusion,
    equal_error_rate,
    percentile,
    precision_recall_f1,
    prf_for,
    sweep,
    wilson_interval,
)


def test_precision_recall_f1_by_hand() -> None:
    precision, recall, f1 = precision_recall_f1(8, 2, 4)
    assert precision == pytest.approx(0.8)
    assert recall == pytest.approx(8 / 12)
    assert f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_zero_division_gives_zero() -> None:
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert precision_recall_f1(0, 3, 0) == (0.0, 0.0, 0.0)


def test_confusion_rows_are_expected_columns_are_predicted() -> None:
    pairs = [("a", "a"), ("a", "b"), ("b", "b"), ("b", "b"), ("b", "zzz")]
    matrix = confusion(pairs, ["a", "b"])
    assert matrix == {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 2}}


def test_prf_for_counts() -> None:
    pairs = [("s", "s"), ("s", "n"), ("n", "s"), ("n", "n")]
    precision, recall, _f1, tp, fp, fn = prf_for(pairs, "s")
    assert (tp, fp, fn) == (1, 1, 1)
    assert precision == recall == 0.5


@pytest.mark.parametrize("q", [0, 5, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_statistics_inclusive(q: int) -> None:
    values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 0.2, 7.7]
    expected = (
        statistics.quantiles(values, n=100, method="inclusive")[q - 1]
        if 0 < q < 100
        else (min(values) if q == 0 else max(values))
    )
    assert percentile(values, q) == pytest.approx(expected)


def test_percentile_empty_and_single() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([2.0], 95) == 2.0


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_sweep_recall_never_rises_with_the_cutoff() -> None:
    scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2]
    truths = [False, False, True, True, True, False]
    rows = sweep(scores, truths, [0.0, 0.25, 0.5, 0.75, 1.0])
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert rows[0]["recall"] == 1.0
    assert rows[-1]["recall"] == 0.0


def test_equal_error_rate_perfect_and_degenerate() -> None:
    assert equal_error_rate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])[0] == 0.0
    assert equal_error_rate([1, 1], [0.5, 0.6]) == (0.0, 0.0)
    assert equal_error_rate([0, 1], [0.9, 0.1])[0] == 1.0
