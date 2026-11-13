"""Tests for the reply message."""

import pytest

from scambot.agent.analyze import Analysis
from scambot.agent.reply import (
    Explanation,
    ReplyInputs,
    choose_reply_language,
    compose_reply,
)
from scambot.core.config import AppConfig
from scambot.core.contracts import CheckerResult, Evidence, Verdict
from scambot.core.llm import LLMClient, LLMError
from scambot.core.locales import Locales
from tests.fakes import FakeLLMClient, SlowLLMClient

SCAM = CheckerResult(
    checker="analyze_scam_patterns",
    status="ok",
    signal="scam",
    confidence=0.8,
    duration_ms=5,
    evidence=[Evidence(summary='urgency: "immediately"', source="rules")],
    details={"signals": ["electricity_disconnection", "urgency", "phone_number", "bogus"]},
)
SCAM_VERDICT = Verdict(
    label="likely_scam", band="high", score=94, deciding_checker="analyze_scam_patterns"
)
CLEAN = CheckerResult(
    checker="analyze_scam_patterns", status="ok", signal="none", confidence=0.9, duration_ms=5
)
CLEAN_VERDICT = Verdict(
    label="no_issues", band="low", score=7, deciding_checker="analyze_scam_patterns"
)
UNSURE_VERDICT = Verdict(
    label="could_not_verify", band="unknown", score=50, deciding_checker="analyze_scam_patterns"
)


async def compose(
    app_config: AppConfig,
    locales: Locales,
    llm: LLMClient,
    inputs: ReplyInputs,
) -> tuple[str, str]:
    reply = await compose_reply(
        inputs,
        llm=llm,
        locales=locales,
        config=app_config.thresholds.reply,
        timeout_s=0.05,
    )
    return reply.text, reply.explanation_source


@pytest.mark.parametrize("language", ["en", "hi"])
async def test_template_scam_reply_layout(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient, language: str
) -> None:
    texts = locales.get(language)

    text, source = await compose(
        app_config, locales, fake_llm, ReplyInputs(SCAM_VERDICT, [SCAM], language, use_llm=True)
    )

    parts = text.split("\n\n")
    assert parts[0] == f"🔴 {texts.verdict.bands.high} · {texts.verdict.labels.likely_scam}"
    signals = texts.verdict.signals
    listed = ", ".join([signals.electricity_disconnection, signals.urgency, signals.phone_number])
    assert parts[1] == texts.verdict.explanations.likely_scam.replace("{signals}", listed)
    assert parts[-1] == texts.disclaimer
    assert len(parts) == 3
    assert source == "template"


@pytest.mark.parametrize(
    ("verdict", "emoji"),
    [
        (Verdict(label="old_or_out_of_context", band="medium", score=60), "🟠"),
        (CLEAN_VERDICT, "🟢"),
        (Verdict(label="could_not_verify", band="unknown", score=50), "⚪"),
    ],
)
async def test_band_emoji(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient, verdict: Verdict, emoji: str
) -> None:
    text, _ = await compose(
        app_config, locales, fake_llm, ReplyInputs(verdict, [], "en", use_llm=False)
    )

    assert text.startswith(f"{emoji} ")
    assert getattr(locales.get("en").verdict.explanations, verdict.label) in text


async def test_generic_signals_when_none_are_known(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    bare = SCAM.model_copy(update={"details": {}})

    text, _ = await compose(
        app_config, locales, fake_llm, ReplyInputs(SCAM_VERDICT, [bare], "en", use_llm=False)
    )

    assert locales.get("en").verdict.signals_generic in text


async def test_llm_explanation_is_used(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    fake_llm.responses["explanation"] = Explanation(
        explanation="  This message threatens to cut your power and asks you to call. Do not call. "
    )

    text, source = await compose(
        app_config, locales, fake_llm, ReplyInputs(SCAM_VERDICT, [SCAM], "en", use_llm=True)
    )

    assert source == "llm"
    assert text.split("\n\n")[1] == (
        "This message threatens to cut your power and asks you to call. Do not call."
    )
    ((name, variables),) = fake_llm.calls
    assert name == "explanation"
    assert '"label": "likely_scam"' in variables["verdict_json"]
    assert '"confidence": 0.8' in variables["results_json"]
    assert "http" not in variables["results_json"]


@pytest.mark.parametrize(
    "explanation",
    [
        "Visit https://sbi.co.in to check.",
        "Go to www.example.org now.",
        "Check sbi-kyc.online first.",
        "Call 98765 43210 to confirm.",
        "Our checker found a pattern.",
        "The LLM thinks this is fine.",
        "   ",
        "x" * 600,
    ],
)
async def test_bad_explanations_fall_back_to_the_template(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient, explanation: str
) -> None:
    fake_llm.responses["explanation"] = Explanation(explanation=explanation)

    text, source = await compose(
        app_config, locales, fake_llm, ReplyInputs(SCAM_VERDICT, [SCAM], "en", use_llm=True)
    )

    assert source == "template"
    assert explanation.strip() not in text or not explanation.strip()


@pytest.mark.parametrize(
    ("explanation", "source"),
    [
        ("We found no warning signs, and the message is genuine.", "template"),
        ("यह संदेश असली है।", "template"),
        ("We found no warning signs. This does not prove it is true.", "llm"),
        ("No warning signs found; this is no proof that it is real.", "llm"),
        ("कोई चेतावनी नहीं मिली, पर यह साबित नहीं करता कि यह सच है।", "llm"),
        ("It is not fake, it is genuine.", "template"),
        ("We did not find problems and the message is authentic.", "template"),
        ("This isn't a scam but it is true.", "template"),
        ("The message is true. It is not a scam, and it is real.", "template"),
        ("We found no warning signs, but stay careful.", "llm"),
        ("There is no doubt it is genuine.", "template"),
        ("Without a doubt the message is real.", "template"),
        ("इसमें कोई शक नहीं कि यह सच है।", "template"),
    ],
)
async def test_genuine_terms_are_rejected_without_confirmation(
    app_config: AppConfig,
    locales: Locales,
    fake_llm: FakeLLMClient,
    explanation: str,
    source: str,
) -> None:
    fake_llm.responses["explanation"] = Explanation(explanation=explanation)

    _, used = await compose(
        app_config, locales, fake_llm, ReplyInputs(UNSURE_VERDICT, [CLEAN], "en", use_llm=True)
    )

    assert used == source


async def test_genuine_terms_allowed_when_source_confirmed(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    fake_llm.responses["explanation"] = Explanation(explanation="The original photo is authentic.")
    verdict = UNSURE_VERDICT.model_copy(update={"source_confirmed": True})

    _, used = await compose(
        app_config, locales, fake_llm, ReplyInputs(verdict, [CLEAN], "en", use_llm=True)
    )

    assert used == "llm"


@pytest.mark.parametrize("language", ["en", "hi"])
async def test_no_issues_always_says_it_is_not_proof(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient, language: str
) -> None:
    fake_llm.responses["explanation"] = Explanation(explanation="All good, nothing to worry about.")

    text, used = await compose(
        app_config, locales, fake_llm, ReplyInputs(CLEAN_VERDICT, [CLEAN], language, use_llm=True)
    )

    assert used == "template"
    assert locales.get(language).verdict.explanations.no_issues in text
    assert "not proof" in locales.get("en").verdict.explanations.no_issues
    assert "प्रमाण नहीं" in locales.get("hi").verdict.explanations.no_issues


async def test_llm_error_and_timeout_fall_back(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    fake_llm.responses["explanation"] = LLMError("down")
    inputs = ReplyInputs(SCAM_VERDICT, [SCAM], "en", use_llm=True)

    _, failed = await compose(app_config, locales, fake_llm, inputs)
    _, slow = await compose(app_config, locales, SlowLLMClient(), inputs)

    assert (failed, slow) == ("template", "template")


async def test_use_llm_false_makes_no_call(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    fake_llm.responses["explanation"] = Explanation(explanation="Fine words.")

    _, source = await compose(
        app_config, locales, fake_llm, ReplyInputs(SCAM_VERDICT, [SCAM], "en", use_llm=False)
    )

    assert source == "template"
    assert fake_llm.calls == []


async def test_unavailable_sentence_and_sources(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    def result(checker: str, *urls: str, status: str = "ok") -> CheckerResult:
        return CheckerResult.model_validate(
            {
                "checker": checker,
                "status": status,
                "signal": "none",
                "confidence": 0.5,
                "duration_ms": 1,
                "evidence": [{"summary": "s", "url": url} for url in urls],
            }
        )

    results = [
        result("check_provenance", "https://a.example/1", "ftp://x.example", "https://b.example/2"),
        result("fact_check_claim", "https://c.example/3", "https://a.example/1"),
        result("detect_deepfake", status="error"),
        result("check_caption_context", "https://d.example/4"),
    ]
    verdict = Verdict(
        label="false_misleading",
        band="high",
        score=90,
        deciding_checker="fact_check_claim",
        unavailable=["detect_deepfake", "detect_voice_clone"],
    )

    text, _ = await compose(
        app_config, locales, fake_llm, ReplyInputs(verdict, results, "hi", use_llm=False)
    )

    hi = locales.get("hi").verdict
    parts = text.split("\n\n")
    checks = f"{hi.checkers.detect_deepfake}, {hi.checkers.detect_voice_clone}"
    assert parts[2] == hi.unavailable.replace("{checks}", checks)
    assert parts[3] == (
        f"{hi.sources}:\n• https://c.example/3\n• https://a.example/1\n• https://b.example/2"
    )
    assert parts[4] == locales.get("hi").disclaimer


def analysis(language: str) -> Analysis:
    return Analysis(
        claims=[],
        reply_language=language,
        claimed_date=None,
        claimed_place=None,
        extra_checkers=[],
        reason="",
    )


@pytest.mark.parametrize(
    ("detected", "preferred", "expected"),
    [
        ("hi", "en", "hi"),
        ("en", "hi", "en"),
        ("hi-Latn", "hi-IN", "hi"),
        ("hi-Latn", "en", "en"),
        ("ta", None, "en"),
        (None, "hi", "hi"),
        ("", "ta", "en"),
    ],
)
def test_choose_reply_language(
    locales: Locales, detected: str | None, preferred: str | None, expected: str
) -> None:
    found = analysis(detected) if detected is not None else None

    assert choose_reply_language(found, preferred, locales) == expected


def voice_result(
    signal: str = "likely_synthetic", status: str = "ok", confidence: float = 0.8
) -> CheckerResult:
    return CheckerResult.model_validate(
        {
            "checker": "detect_voice_clone",
            "status": status,
            "signal": signal,
            "confidence": confidence,
            "duration_ms": 5,
            "error": "inference_failed" if status == "error" else None,
        }
    )


@pytest.mark.parametrize("language", ["en", "hi"])
@pytest.mark.parametrize("use_llm", [False, True])
async def test_voice_note_is_added_after_the_explanation(
    app_config: AppConfig,
    locales: Locales,
    fake_llm: FakeLLMClient,
    language: str,
    use_llm: bool,
) -> None:
    texts = locales.get(language)
    fake_llm.responses["explanation"] = Explanation(
        explanation="This message asks for money in a hurry and for a code."
        if language == "en"
        else "यह संदेश जल्दी में पैसे और कोड माँगता है।"
    )
    inputs = ReplyInputs(SCAM_VERDICT, [SCAM, voice_result()], language, use_llm=use_llm)

    text, source = await compose(app_config, locales, fake_llm, inputs)

    parts = text.split("\n\n")
    assert parts[2] == texts.verdict.voice_note
    assert parts[-1] == texts.disclaimer
    assert len(parts) == 4
    assert source == ("llm" if use_llm else "template")
    assert "genuine" not in text.lower()


@pytest.mark.parametrize(
    "result",
    [
        voice_result("none", confidence=0.9),
        voice_result("none", "error", 0.0),
        voice_result("none", "inconclusive", 0.0),
    ],
)
async def test_voice_note_is_absent_without_a_synthetic_signal(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient, result: CheckerResult
) -> None:
    text, _ = await compose(
        app_config,
        locales,
        fake_llm,
        ReplyInputs(SCAM_VERDICT, [SCAM, result], "en", use_llm=False),
    )

    assert locales.get("en").verdict.voice_note not in text


async def test_voice_note_comes_before_the_unavailable_line(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    verdict = Verdict(
        label="possibly_ai",
        band="medium",
        score=60,
        deciding_checker="detect_voice_clone",
        unavailable=["fact_check_claim"],
    )

    text, _ = await compose(
        app_config,
        locales,
        fake_llm,
        ReplyInputs(verdict, [voice_result()], "en", use_llm=False),
    )

    parts = text.split("\n\n")
    assert parts[2] == locales.get("en").verdict.voice_note
    assert "fact-check" in parts[3].lower() or parts[3].startswith("Some checks")
    assert len(text) < 1500


async def test_voice_note_is_outside_the_summary_limit(
    app_config: AppConfig, locales: Locales, fake_llm: FakeLLMClient
) -> None:
    fake_llm.responses["explanation"] = Explanation(explanation="x" * 400)
    inputs = ReplyInputs(SCAM_VERDICT, [SCAM, voice_result()], "en", use_llm=True)

    text, source = await compose(app_config, locales, fake_llm, inputs)

    assert source == "llm"  # headline + 400 fits; the fixed voice note is not counted
    assert locales.get("en").verdict.voice_note in text
