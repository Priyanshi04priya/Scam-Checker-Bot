"""The committed seed data is well formed, honest and balanced."""

import re
from collections import Counter
from pathlib import Path

import pytest
from eval.schema import Item, devanagari_ratio, load_dir, load_items

from scambot.core.config import load_config

SEED = Path(__file__).parents[2] / "eval" / "data" / "seed"
TEMPLATES = Path(__file__).parents[2] / "eval" / "data" / "templates"
URL_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)
OK_HOST = re.compile(r"(^|\.)example\.(com|org|net|test|invalid)$|\.(invalid|test)$")


@pytest.fixture(scope="module")
def items() -> list[Item]:
    return load_dir(SEED)


def of(items: list[Item], suite: str) -> list[Item]:
    return [i for i in items if i.suite == suite]


def test_counts(items: list[Item]) -> None:
    scam = of(items, "scam")
    assert Counter(i.expected_label for i in scam) == {"likely_scam": 30, "no_issues": 30}
    assert len(of(items, "unverifiable")) == 30
    assert len(of(items, "injection")) == 20
    assert len(of(items, "factcheck")) == 10
    assert {i.expected_label for i in of(items, "unverifiable")} == {"could_not_verify"}


def test_languages_and_splits_are_balanced(items: list[Item]) -> None:
    scam = of(items, "scam")
    assert Counter(i.language for i in scam) == {"en": 20, "hi": 20, "hinglish": 20}
    for label in ("likely_scam", "no_issues"):
        for language in ("en", "hi", "hinglish"):
            group = [i for i in scam if i.expected_label == label and i.language == language]
            assert {i.split for i in group} == {"dev", "test"}
            assert abs(sum(i.split == "dev" for i in group) - len(group) / 2) <= 1
    for suite in ("unverifiable", "injection", "factcheck"):
        assert {i.split for i in of(items, suite)} == {"dev", "test"}


def test_every_row_has_provenance_and_unique_id(items: list[Item]) -> None:
    assert len({i.id for i in items}) == len(items)
    assert all(i.source_note.strip() for i in items)
    for i in items:
        if i.suite != "scam":
            assert i.synthetic, i.id


def test_injection_canaries_are_unique_and_present(items: list[Item]) -> None:
    injection = of(items, "injection")
    canaries = [i.canary for i in injection]
    assert len(set(canaries)) == 20
    assert all(i.canary and i.canary in i.text for i in injection)


def test_only_reserved_hosts_appear(items: list[Item]) -> None:
    for i in items:
        for host in URL_RE.findall(i.text):
            assert OK_HOST.search(host.lower()), (i.id, host)


def test_hindi_rows_are_devanagari_and_others_are_not(items: list[Item]) -> None:
    for i in items:
        if i.language == "hi":
            assert devanagari_ratio(i.text) > 0.3, i.id
        else:
            assert devanagari_ratio(i.text) == 0, i.id


def test_no_real_fact_checker_names_or_domains(items: list[Item]) -> None:
    sources = load_config().sources
    banned = {s.site.lower() for s in sources.sources} | {s.name.lower() for s in sources.sources}
    articles = (SEED / "synthetic_articles.jsonl").read_text("utf-8")
    text = (" ".join(i.text + i.source_note + (i.expected_source or "") for i in items)).lower()
    for blob in (text, articles.lower()):
        for name in banned:
            assert name not in blob, name


def test_synthetic_articles_match_factcheck_items(items: list[Item]) -> None:
    articles = (SEED / "synthetic_articles.jsonl").read_text("utf-8")
    for item in of(items, "factcheck"):
        assert item.expected_source
        assert item.expected_source in articles
        assert "example.invalid" in item.expected_source


def test_replay_file_is_small() -> None:
    assert (SEED / "replay_offline.json").stat().st_size < 200_000


def test_templates_are_marked_placeholders() -> None:
    for path in TEMPLATES.glob("*.template.jsonl"):
        (only,) = load_items(path)
        assert only.source_note.startswith("SYNTHETIC-PLACEHOLDER")
        assert only.synthetic
