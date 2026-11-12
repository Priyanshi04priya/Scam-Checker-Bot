"""Reading a publisher's rating text ("False", "Missing Context", "गलत") as a polarity."""

import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache

from scambot.core.config import RatingRule
from scambot.core.contracts import RatingPolarity


def _normalise(text: str) -> str:
    """NFC and case-folded, so a nukta letter typed as one or two code points matches."""
    return unicodedata.normalize("NFC", text).casefold()


@lru_cache(maxsize=256)
def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(_normalise(term))
    if term.isascii():
        return re.compile(rf"\b{escaped}\b")
    return re.compile(escaped)


def rating_polarity(rating: str | None, rules: Sequence[RatingRule]) -> RatingPolarity:
    """The first rule (in file order) with a term found in ``rating``; ``unrated`` otherwise.

    English terms match whole words, so "untrue" never counts as "true"; other terms match
    anywhere, because Hindi words carry attached particles.
    """
    if not rating:
        return "unrated"
    text = _normalise(rating)
    for rule in rules:
        if any(_term_pattern(term).search(text) for term in rule.terms):
            return rule.polarity
    return "unrated"
