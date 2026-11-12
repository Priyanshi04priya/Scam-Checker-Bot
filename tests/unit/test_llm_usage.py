"""Tests for the token usage ledger and cost."""

import asyncio
import logging
from decimal import Decimal

import pytest

from scambot.core.config import Price
from scambot.core.llm import LLMUsage, cost_usd, record_usage, usage_scope


def usage(model: str = "claude-opus-5-5", tokens_in: int = 1000, tokens_out: int = 100) -> LLMUsage:
    return LLMUsage(call="analyze", model=model, input_tokens=tokens_in, output_tokens=tokens_out)


def test_scope_collects_and_isolates() -> None:
    record_usage(usage())  # outside any scope: ignored
    with usage_scope() as outer:
        record_usage(usage())
        with usage_scope() as inner:
            record_usage(usage("claude-haiku-5-5"))
        record_usage(usage())

    assert [u.model for u in inner] == ["claude-haiku-5-5"]
    assert len(outer) == 2


async def test_child_tasks_share_the_scope() -> None:
    async def child() -> None:
        record_usage(usage())

    with usage_scope() as usages:
        await asyncio.gather(child(), child())

    assert len(usages) == 2


def test_cost(caplog: pytest.LogCaptureFixture) -> None:
    prices = {"claude-opus-5-5": Price(input=Decimal("4.00"), output=Decimal("20.00"))}
    caplog.set_level(logging.WARNING)

    total = cost_usd([usage(), usage(), usage("mystery-model")], prices)

    # 2 x (1000 x 4 + 100 x 20) / 1,000,000
    assert total == Decimal("0.012000")
    assert cost_usd([], prices) == Decimal("0.000000")
