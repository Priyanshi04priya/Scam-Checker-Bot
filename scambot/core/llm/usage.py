"""Token usage ledger for one job, and its cost."""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from scambot.core.logging import get_logger

_log = get_logger(component="llm")

_PLACES = Decimal("0.000001")


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class TokenPrice(Protocol):
    """Price in US dollars per million tokens."""

    @property
    def input(self) -> Decimal: ...

    @property
    def output(self) -> Decimal: ...


_ledger: ContextVar[list[LLMUsage] | None] = ContextVar("llm_usage", default=None)


@contextmanager
def usage_scope() -> Iterator[list[LLMUsage]]:
    """Collect every usage recorded inside the block (child tasks share the same list)."""
    usages: list[LLMUsage] = []
    token = _ledger.set(usages)
    try:
        yield usages
    finally:
        _ledger.reset(token)


def record_usage(usage: LLMUsage) -> None:
    """Add ``usage`` to the current scope; does nothing outside a scope."""
    usages = _ledger.get()
    if usages is not None:
        usages.append(usage)


def cost_usd(usages: Iterable[LLMUsage], prices: Mapping[str, TokenPrice]) -> Decimal:
    """Total price of ``usages``; a model without a price counts as free (and is logged)."""
    total = Decimal(0)
    unknown: set[str] = set()
    for usage in usages:
        price = prices.get(usage.model)
        if price is None:
            unknown.add(usage.model)
            continue
        total += usage.input_tokens * price.input + usage.output_tokens * price.output
    for model in sorted(unknown):
        _log.warning("llm_price_unknown", model=model)
    return (total / 1_000_000).quantize(_PLACES)
