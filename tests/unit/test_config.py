"""Tests for the behaviour configuration files and their validation."""

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml

from scambot.core.config import AppConfig, load_config
from scambot.core.contracts import CheckerName, JobKind
from scambot.core.settings import CONFIG_DIR

Mutation = Callable[[dict[str, Any]], None]


def test_shipped_files_load(app_config: AppConfig) -> None:
    routing = app_config.routing
    assert set(routing.routes) == set(get_args(JobKind))
    assert set(routing.checkers) == set(get_args(CheckerName))
    enabled = [name for name, checker in routing.checkers.items() if checker.enabled]
    assert enabled == ["analyze_scam_patterns", "fact_check_claim"]
    assert routing.checkers["check_caption_context"].depends_on == ["check_provenance"]
    assert app_config.thresholds.pipeline.deadline_s == 90
    assert app_config.thresholds.scam.signal_min == 0.40
    assert "claude-opus-5-5" in app_config.thresholds.llm_prices_usd_per_mtok
    assert "bit.ly" in app_config.scam_patterns.shorteners
    assert app_config.thresholds.factcheck.rrf_k == 60
    assert app_config.thresholds.factcheck.relevance_timeout_s == 12
    assert app_config.thresholds.ingest.max_page_bytes == 3_000_000
    sources = app_config.sources
    assert {source.id for source in sources.sources} >= {"pib", "altnews", "factly", "boom"}
    assert [rule.polarity for rule in sources.ratings] == ["mixed", "misleading", "false", "true"]
    assert sources.by_id("factly") is not None
    assert sources.by_id("nobody") is None
    altnews = sources.by_host("www.altnews.in")
    assert altnews is not None
    assert altnews.id == "altnews"  # the api source wins over the Hindi feed-only entry
    assert sources.by_host("notaltnews.in") is None


def _copy_with(tmp_path: Path, name: str, mutate: Mutation) -> Path:
    for path in CONFIG_DIR.glob("*.yaml"):
        shutil.copy(path, tmp_path / path.name)
    target = tmp_path / name
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    mutate(data)
    target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return tmp_path


def _set(path: str, value: object) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        node: Any = data
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if value is _DELETE:
            del node[last]
        else:
            node[last] = value

    return mutate


_DELETE = object()


def _move_always_first(data: dict[str, Any]) -> None:
    data["rules"].insert(0, data["rules"].pop())


def _add_second_always(data: dict[str, Any]) -> None:
    data["rules"].insert(0, dict(data["rules"][-1], id="early"))


@pytest.mark.parametrize(
    ("file", "mutate"),
    [
        ("routing.yaml", _set("routes.text", ["analyze_scam_patterns", "spot_the_robot"])),
        ("routing.yaml", _set("routes.audio", _DELETE)),
        ("routing.yaml", _set("routes.text", ["analyze_scam_patterns", "analyze_scam_patterns"])),
        ("routing.yaml", _set("checkers.detect_deepfake", _DELETE)),
        ("routing.yaml", _set("checkers.check_provenance.depends_on", ["check_caption_context"])),
        ("routing.yaml", _set("checkers.check_provenance.depends_on", ["check_provenance"])),
        ("routing.yaml", _set("checkers.analyze_scam_patterns.timeout_s", 0)),
        ("verdict_rules.yaml", _move_always_first),
        ("verdict_rules.yaml", _add_second_always),
        ("verdict_rules.yaml", _set("rules.1.id", "scam")),
        ("verdict_rules.yaml", _set("rules.0.when.always", True)),
        ("verdict_rules.yaml", _set("bands.high.min", 101)),
        ("verdict_rules.yaml", _set("bands.medium", {"min": 69, "max": 40})),
        ("verdict_rules.yaml", _set("required.voice", _DELETE)),
        ("verdict_rules.yaml", _set("rules.0.when.any_of.0.min_confidence", 1.5)),
        ("thresholds.yaml", _set("scam.signal_min", 2)),
        ("thresholds.yaml", _set("pipeline.surprise", 1)),
        ("scam_patterns.yaml", _set("templates.kyc_update", _DELETE)),
        ("scam_patterns.yaml", _set("templates.made_up", {"weight": 0.5, "patterns": {}})),
        ("scam_patterns.yaml", _set("templates.kyc_update.patterns.en", ["(unclosed"])),
        ("scam_patterns.yaml", _set("templates.kyc_update.patterns.hinglish", _DELETE)),
        ("scam_patterns.yaml", _set("signals.urgency.patterns.hi", [])),
        ("scam_patterns.yaml", _set("signals.threat.weight", 1.0)),
        ("scam_patterns.yaml", _set("entity_signals.upi_id", _DELETE)),
        ("thresholds.yaml", _set("factcheck.rrf_k", 0)),
        ("thresholds.yaml", _set("media.download_timeout_s", 60)),
        ("thresholds.yaml", _set("media.ocr.languages", "eng hin")),
        ("thresholds.yaml", _set("media.ocr.min_word_conf", 101)),
        ("thresholds.yaml", _set("media.stt", _DELETE)),
        ("thresholds.yaml", _set("ingest", _DELETE)),
        ("sources.yaml", _set("sources.1.id", "pib")),
        ("sources.yaml", _set("sources.1.feeds", ["https://www.example.com/feed/"])),
        ("sources.yaml", _set("sources.1.feeds", ["http://www.altnews.in/feed/"])),
        ("sources.yaml", lambda data: data["ratings"].pop()),
        ("sources.yaml", _set("ratings.3.polarity", "false")),
        ("sources.yaml", _set("ratings.0.terms", ["mixed", " "])),
        ("sources.yaml", _set("sources.0.site", "PIB Gov")),
        ("sources.yaml", _set("sources.0.language", "english")),
    ],
)
def test_invalid_copy_fails_loading(tmp_path: Path, file: str, mutate: Mutation) -> None:
    directory = _copy_with(tmp_path, file, mutate)

    with pytest.raises(ValueError, match=rf"invalid config file {re.escape(file)}"):
        load_config(directory)


def test_missing_file_fails_loading(tmp_path: Path) -> None:
    directory = _copy_with(tmp_path, "routing.yaml", lambda _data: None)
    (directory / "thresholds.yaml").unlink()

    with pytest.raises(ValueError, match=r"thresholds\.yaml"):
        load_config(directory)


def test_media_thresholds_load(app_config: AppConfig) -> None:
    media = app_config.thresholds.media

    assert (media.download_timeout_s, media.preprocess_timeout_s) == (15, 35)
    assert media.ocr.languages == "eng+hin"
    assert media.stt.window_s == 60
    assert media.keyframes.max_keyframes == 6
    assert media.sweep_older_than_min == 45
