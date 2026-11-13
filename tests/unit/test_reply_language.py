"""Which language and style a reply uses: explicit choice, message language, app language."""

from pathlib import Path

import pytest

from scambot.agent.analyze import Analysis
from scambot.agent.reply import (
    ReplyInputs,
    _llm_variables,
    choose_reply_language,
    compose_reply,
    reply_style,
)
from scambot.core.config import AppConfig
from scambot.core.contracts import Verdict
from scambot.core.llm import PromptLibrary
from scambot.core.locales import Locales, load_locales
from tests.fakes import FakeLLMClient


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
    ("detected", "app", "chosen", "language", "style"),
    [
        ("en", "en", "hi", "hi", None),  # an explicit choice beats the message language
        ("hi", "en", None, "hi", None),  # no choice: the message language
        ("hi-Latn", "en", None, "en", "hi-Latn"),  # Hinglish detected: English text, Hinglish style
        ("hi-Latn", "en", "en", "en", None),  # an explicit choice beats detection
        ("hi-Latn", "en", "hi-Latn", "en", "hi-Latn"),
        ("xx", "hi-IN", None, "hi", None),  # app language
        ("xx", "ta", None, "en", None),  # unknown language
    ],
)
def test_precedence(
    locales: Locales, detected: str, app: str, chosen: str | None, language: str, style: str | None
) -> None:
    found = analysis(detected)
    assert choose_reply_language(found, app, locales, chosen) == language
    assert reply_style(found, chosen) == style


def test_resolving_hinglish_gives_english(locales: Locales) -> None:
    assert locales.resolve("hi-Latn") == "en"
    assert locales.resolve("hi-IN") == "hi"


def test_selectable_languages(locales: Locales) -> None:
    assert locales.selectable() == ["en", "hi", "hi-Latn"]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("English", "en"),
        (" hi ", "hi"),
        ("Hindi", "hi"),
        ("हिन्दी", "hi"),
        ("hinglish", "hi-Latn"),
        ("HI-LATN", "hi-Latn"),
        ("klingon", None),
        ("", None),
    ],
)
def test_normalise_choice(locales: Locales, text: str, code: str | None) -> None:
    assert locales.normalise_choice(text) == code


def test_an_added_locale_file_becomes_selectable(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "config" / "locales"
    for name in ("en.yaml", "hi.yaml"):
        (tmp_path / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "bn.yaml").write_text((source / "en.yaml").read_text(encoding="utf-8"), "utf-8")

    locales = load_locales(tmp_path)

    assert locales.selectable() == ["en", "bn", "hi", "hi-Latn"]
    assert locales.normalise_choice("BN") == "bn"


async def test_hinglish_reaches_the_prompt_and_the_template_is_english(
    locales: Locales, prompts: PromptLibrary, app_config: AppConfig
) -> None:
    verdict = Verdict(label="no_issues", band="low", score=10)
    inputs = ReplyInputs(verdict, [], "en", use_llm=False, style="hi-Latn")

    assert _llm_variables(inputs)["language"] == "hi-Latn"
    assert _llm_variables(ReplyInputs(verdict, [], "en", use_llm=False))["language"] == "en"
    reply = await compose_reply(
        inputs,
        llm=FakeLLMClient(prompts),
        locales=locales,
        config=app_config.thresholds.reply,
        timeout_s=1,
    )
    assert reply.text.endswith(locales.get("en").disclaimer)
