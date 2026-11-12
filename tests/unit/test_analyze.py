"""Tests for request analysis."""

import datetime as dt

import pytest

from scambot.agent.analyze import Analysis, analyze
from scambot.core.contracts import Preprocessed
from scambot.core.llm import LLMRefusalError
from tests.fakes import FakeLLMClient, SlowLLMClient

TODAY = dt.date(2026, 11, 12)


def analysis(**overrides: object) -> Analysis:
    data: dict[str, object] = {
        "claims": [],
        "reply_language": "en",
        "claimed_date": None,
        "claimed_place": None,
        "extra_checkers": [],
        "reason": "",
    }
    return Analysis.model_validate(data | overrides)


async def run(fake_llm: FakeLLMClient, text: str = "Some message") -> Analysis | None:
    return await analyze(
        Preprocessed(kind="text", text=text), llm=fake_llm, received_on=TODAY, timeout_s=1
    )


async def test_variables_and_normalisation(fake_llm: FakeLLMClient) -> None:
    fake_llm.responses["analyze"] = analysis(
        claims=["  one  ", "", "two", "x" * 400, "four"],
        reply_language="hi-Latn",
        claimed_place="  " + "P" * 150,
        extra_checkers=["fact_check_claim", "fact_check_claim", "analyze_scam_patterns"],
    )

    result = await run(fake_llm)

    assert result is not None
    assert result.claims == ["one", "two", "x" * 300]
    assert result.reply_language == "hi-Latn"
    assert result.claimed_place == "P" * 100
    assert result.extra_checkers == ["fact_check_claim", "analyze_scam_patterns"]
    assert fake_llm.calls == [
        ("analyze", {"content": "Some message", "kind": "text", "received_on": "2026-11-12"})
    ]


@pytest.mark.parametrize(
    ("raw", "kept"), [("en", "en"), ("hi", "hi"), ("Hindi", ""), ("en-US-x", ""), (" hi ", "hi")]
)
async def test_reply_language_is_validated(fake_llm: FakeLLMClient, raw: str, kept: str) -> None:
    fake_llm.responses["analyze"] = analysis(reply_language=raw, claimed_place="  ")

    result = await run(fake_llm)

    assert result is not None
    assert result.reply_language == kept
    assert result.claimed_place is None


async def test_failure_gives_none(fake_llm: FakeLLMClient) -> None:
    fake_llm.responses["analyze"] = LLMRefusalError("refused")

    assert await run(fake_llm) is None


async def test_timeout_gives_none() -> None:
    result = await analyze(
        Preprocessed(kind="text", text="x"), llm=SlowLLMClient(), received_on=TODAY, timeout_s=0.01
    )

    assert result is None


async def test_empty_text_makes_no_call(fake_llm: FakeLLMClient) -> None:
    assert await run(fake_llm, "  ") is None
    assert fake_llm.calls == []
