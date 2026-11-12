"""Tests for reading publisher ratings."""

import pytest

from scambot.core.config import AppConfig
from scambot.core.contracts import RatingPolarity
from scambot.factcheck_index.ratings import rating_polarity


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        ("False", "false"),
        ("Fake", "false"),
        ("Mostly False", "false"),
        ("Pants on Fire!", "false"),
        ("Half True", "mixed"),
        ("Partly false", "mixed"),
        ("Missing Context", "misleading"),
        ("Not true", "false"),
        ("untrue", "false"),
        ("True", "true"),
        ("गलत", "false"),
        ("फर्जी दावा", "false"),
        ("सच नहीं", "false"),
        ("भ्रामक", "misleading"),
        ("\u095a\u0932\u0924", "false"),  # ग़लत with the precomposed nukta letter
        ("Not correct", "false"),
        ("Not entirely true", "mixed"),
        ("Unproven", "unrated"),
        (None, "unrated"),
        ("", "unrated"),
    ],
)
def test_rating_polarity(
    app_config: AppConfig, rating: str | None, expected: RatingPolarity
) -> None:
    assert rating_polarity(rating, app_config.sources.ratings) == expected
