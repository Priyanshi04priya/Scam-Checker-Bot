"""Tests for reciprocal rank fusion and record fusion."""

import pytest

from scambot.factcheck_index.records import ArticleRecord
from scambot.factcheck_index.search import fuse, reciprocal_rank_fusion


def record(url: str, rating: str | None = None) -> ArticleRecord:
    return ArticleRecord(source="s", url=url, title="t", claim="c", rating=rating)


def test_scores_add_up_across_lists() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "x", "a"]], k=60))

    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["x"] == pytest.approx(1 / 62)


def test_three_lists_and_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b"], ["c", "b"]], k=60)

    assert [key for key, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61 + 1 / 62)


def test_item_in_one_list_only_and_empty_lists() -> None:
    ((key, score),) = reciprocal_rank_fusion([["a"], []])
    assert key == "a"
    assert score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


def test_ties_are_broken_by_best_rank_then_key() -> None:
    # y and z tie on score and best rank, as do m and n, so the key decides.
    fused = reciprocal_rank_fusion([["z", "y"], ["y", "z"], ["m"], ["n"]], k=60)
    assert [key for key, _ in fused] == ["y", "z", "m", "n"]


def test_repeated_key_in_one_list_counts_once() -> None:
    assert dict(reciprocal_rank_fusion([["a", "a"]]))["a"] == pytest.approx(1 / 61)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)


def test_fuse_dedupes_by_canonical_url_and_keeps_the_rated_record() -> None:
    plain = record("https://Factly.in/laptop/?utm_source=wa#top")
    rated = record("https://factly.in/laptop/", rating="False")
    other = record("https://boomlive.in/x")

    fused = fuse([[plain, other], [rated]], k=60)

    assert len(fused) == 2
    assert fused[0].rating == "False"
    assert fused[1] is other


def test_fuse_keeps_the_first_record_when_both_are_rated() -> None:
    first = record("https://a.in/1", rating="False")
    second = record("https://a.in/1?fbclid=1", rating="True")

    assert fuse([[first], [second]], k=60) == [first]
