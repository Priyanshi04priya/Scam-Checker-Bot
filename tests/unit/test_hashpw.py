"""The password hash tool for the admin login."""

import pytest
from argon2 import PasswordHasher

from scambot.api.admin import hashpw


def test_prints_a_hash_that_verifies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "s3cret")

    assert hashpw.main() == 0

    assert PasswordHasher().verify(capsys.readouterr().out.strip(), "s3cret")


def test_refuses_mismatched_passwords(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["one", "two"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))

    assert hashpw.main() == 1
    assert "$argon2" not in capsys.readouterr().out
