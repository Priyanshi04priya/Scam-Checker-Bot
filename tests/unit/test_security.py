"""Tests for user-ID hashing and secret comparison."""

import hashlib
import hmac

import pytest
from pydantic import SecretStr

from scambot.core.security import secrets_match, user_hash


def test_user_hash_is_hmac_sha256_of_the_id() -> None:
    expected = hmac.new(b"pepper", b"12345", hashlib.sha256).hexdigest()

    assert user_hash(12345, SecretStr("pepper")) == expected
    assert user_hash("12345", SecretStr("pepper")) == expected


def test_user_hash_depends_on_the_pepper() -> None:
    assert user_hash(1, SecretStr("a")) != user_hash(1, SecretStr("b"))
    assert user_hash(1, SecretStr("a")) != user_hash(2, SecretStr("a"))


@pytest.mark.parametrize(
    ("received", "expected", "match"),
    [
        ("token", "token", True),
        ("token", "other", False),
        ("tok", "token", False),
        (None, "token", False),
        ("", "", False),
        (None, "", False),
    ],
)
def test_secrets_match(received: str | None, expected: str, match: bool) -> None:
    assert secrets_match(received, SecretStr(expected)) is match
