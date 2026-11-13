"""Pure metric functions: confusion matrices, precision/recall, percentiles, intervals."""

import math
from collections import Counter
from collections.abc import Sequence


def confusion(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """Rows are the expected label, columns the predicted one; pairs are (expected, predicted)."""
    matrix = {row: dict.fromkeys(labels, 0) for row in labels}
    for expected, predicted in pairs:
        if expected in matrix and predicted in matrix[expected]:
            matrix[expected][predicted] += 1
    return matrix


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1; a zero denominator gives 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) by linear interpolation between closest ranks.

    This is the "inclusive" method: ``percentile(v, 50)`` is the median and the result always
    lies inside ``[min(v), max(v)]``. An empty sequence gives 0.0.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(max(q, 0.0), 100.0) / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def prf_for(
    pairs: Sequence[tuple[str, str]], positive: str
) -> tuple[float, float, float, int, int, int]:
    """Precision, recall, F1 and the tp/fp/fn counts for one positive label."""
    counts = Counter(
        ("tp" if e == positive and p == positive else "fp" if p == positive else "fn")
        for e, p in pairs
        if positive in (e, p)
    )
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return (*precision_recall_f1(tp, fp, fn), tp, fp, fn)


def sweep(
    scores: Sequence[float], truths: Sequence[bool], cutoffs: Sequence[float]
) -> list[dict[str, float]]:
    """Precision, recall and F1 for each cut-off (a score at or above it is positive)."""
    rows: list[dict[str, float]] = []
    for cutoff in cutoffs:
        tp = sum(1 for s, t in zip(scores, truths, strict=True) if s >= cutoff and t)
        fp = sum(1 for s, t in zip(scores, truths, strict=True) if s >= cutoff and not t)
        fn = sum(1 for s, t in zip(scores, truths, strict=True) if s < cutoff and t)
        p, r, f = precision_recall_f1(tp, fp, fn)
        rows.append({"cutoff": cutoff, "precision": p, "recall": r, "f1": f})
    return rows


def equal_error_rate(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    """EER and its threshold; labels are 1 for the positive (fake) class."""
    if not labels or len(set(labels)) < 2:
        return 0.0, 0.0
    positives = sum(labels)
    negatives = len(labels) - positives
    best = (2.0, 0.0, 0.0)
    for cutoff in sorted(set(scores)):
        fnr = sum(1 for s, y in zip(scores, labels, strict=True) if y and s < cutoff) / positives
        false_positives = sum(
            1 for s, y in zip(scores, labels, strict=True) if not y and s >= cutoff
        )
        fpr = false_positives / negatives
        gap = abs(fnr - fpr)
        if gap < best[0]:
            best = (gap, (fnr + fpr) / 2, cutoff)
    return best[1], best[2]
