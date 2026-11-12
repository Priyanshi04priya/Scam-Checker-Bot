"""Tests for the checker planner."""

from scambot.agent.analyze import Analysis
from scambot.agent.planner import build_plan
from scambot.core.config import AppConfig, CheckerConfig, RoutingConfig
from scambot.core.contracts import CheckerName, Preprocessed


def analysis(extra: list[CheckerName], claims: list[str] | None = None) -> Analysis:
    return Analysis(
        claims=claims or [],
        reply_language="en",
        claimed_date=None,
        claimed_place=None,
        extra_checkers=extra,
        reason="",
    )


def all_enabled(app_config: AppConfig) -> RoutingConfig:
    checkers = {
        name: config.model_copy(update={"enabled": True})
        for name, config in app_config.routing.checkers.items()
    }
    return app_config.routing.model_copy(update={"checkers": checkers})


TEXT = Preprocessed(kind="text", text="Your KYC expires today")


def test_baseline_text_plan(app_config: AppConfig) -> None:
    plan = build_plan("text", TEXT, None, app_config.routing)

    assert plan.names == ("analyze_scam_patterns",)
    assert plan.checkers[0].timeout_s == 25
    assert plan.skipped == {"fact_check_claim": "disabled"}


def test_extras_are_added_but_never_removed(app_config: AppConfig) -> None:
    routing = all_enabled(app_config)
    image = Preprocessed(kind="image", text="caption", has_media=True)

    plan = build_plan("image", image, analysis(["detect_voice_clone"], ["a claim"]), routing)

    assert plan.names == (
        "check_provenance",
        "check_caption_context",
        "fact_check_claim",
        "analyze_scam_patterns",
        "detect_deepfake",
    )
    assert plan.skipped == {"detect_voice_clone": "no_input"}
    assert plan.checkers[1].depends_on == ("check_provenance",)


def test_extra_for_text(app_config: AppConfig) -> None:
    routing = all_enabled(app_config)

    plan = build_plan("text", TEXT, analysis(["detect_deepfake"], ["claim"]), routing)

    assert plan.names == ("analyze_scam_patterns", "fact_check_claim")
    assert plan.skipped == {"detect_deepfake": "no_input"}


def test_no_input_skips(app_config: AppConfig) -> None:
    routing = all_enabled(app_config)
    voice = Preprocessed(kind="voice", text="", has_media=True)

    plan = build_plan("voice", voice, analysis([]), routing)

    assert plan.names == ("detect_voice_clone",)
    assert plan.skipped == {"analyze_scam_patterns": "no_input", "fact_check_claim": "no_input"}


def test_dependency_not_planned(app_config: AppConfig) -> None:
    routing = all_enabled(app_config)
    checkers = dict(routing.checkers)
    checkers["check_provenance"] = CheckerConfig(timeout_s=20, enabled=False)
    routing = routing.model_copy(update={"checkers": checkers})
    image = Preprocessed(kind="image", has_media=True)

    plan = build_plan("image", image, None, routing)

    assert "check_caption_context" not in plan.names
    assert plan.skipped["check_provenance"] == "disabled"
    assert plan.skipped["check_caption_context"] == "dependency_not_planned"
    assert plan.names == ("detect_deepfake",)
