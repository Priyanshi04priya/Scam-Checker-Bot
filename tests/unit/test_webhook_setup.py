"""Tests for the webhook management commands (a fake bot stands in for Telegram)."""

from types import SimpleNamespace
from typing import Any

import pytest

from scambot.api import polling
from scambot.api.webhook_setup import (
    delete_webhook,
    describe_error,
    set_webhook,
    validate_secret,
    validate_webhook_url,
    webhook_info,
)
from scambot.channels.telegram import ALLOWED_UPDATES
from scambot.core.settings import get_settings

URL = "https://bot.example.com/webhook/telegram"


class FakeBot:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail
        self.session = SimpleNamespace(close=self._close)
        self.closed = False

    async def _close(self) -> None:
        self.closed = True

    async def set_webhook(self, url: str, **kwargs: Any) -> bool:
        if self.fail:
            raise self.fail
        self.calls.append(("set", {"url": url, **kwargs}))
        return True

    async def delete_webhook(self, **kwargs: Any) -> bool:
        self.calls.append(("delete", kwargs))
        return True

    async def get_webhook_info(self) -> Any:
        self.calls.append(("info", {}))
        return SimpleNamespace(url=URL, pending_update_count=2, last_error_message=None)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (URL, True),
        ("https://bot.example.com:8443/webhook/telegram", True),
        ("https://bot.example.com:443/webhook/telegram", True),
        ("http://bot.example.com/webhook/telegram", False),
        ("https://bot.example.com/other", False),
        ("https://bot.example.com/webhook/telegram?x=1", False),
        ("https://bot.example.com/webhook/telegram#frag", False),
        ("https://user:pw@bot.example.com/webhook/telegram", False),
        ("https://bot.example.com:1234/webhook/telegram", False),
        ("https:///webhook/telegram", False),
        ("https://bot.example.com:notaport/webhook/telegram", False),
    ],
)
def test_validate_webhook_url(url: str, ok: bool) -> None:
    assert (validate_webhook_url(url) is None) is ok


def test_validate_secret() -> None:
    assert validate_secret("abc_DEF-123") is None
    assert validate_secret("has space") is not None
    assert validate_secret("") is not None


async def test_set_webhook_always_registers_with_the_secret() -> None:
    bot = FakeBot()

    await set_webhook(bot, URL, "s" * 20)
    await set_webhook(bot, URL, "s" * 20)

    sets = [kwargs for name, kwargs in bot.calls if name == "set"]
    assert len(sets) == 2
    assert sets[0]["secret_token"] == "s" * 20
    assert sets[0]["allowed_updates"] == ALLOWED_UPDATES
    assert sets[0]["drop_pending_updates"] is False


async def test_delete_and_info() -> None:
    bot = FakeBot()

    await delete_webhook(bot, drop_pending=True)
    text = await webhook_info(bot)

    assert ("delete", {"drop_pending_updates": True}) in bot.calls
    assert "pending_update_count: 2" in text
    assert "last_error_message: (none)" in text


def test_error_text_never_contains_the_token() -> None:
    token = "123456:secret-token"
    assert token not in describe_error(RuntimeError(f"https://api.telegram.org/bot{token}/x"))
    assert describe_error(RuntimeError(token)) == "RuntimeError"


def _settings(monkeypatch: pytest.MonkeyPatch, *, secret: str = "w" * 24) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:token-value")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", secret)
    get_settings.cache_clear()


def test_main_set_webhook_registers_and_prints_info(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _settings(monkeypatch)
    bot = FakeBot()
    monkeypatch.setattr(polling, "create_bot", lambda *_args: bot)

    assert polling.main(["--set-webhook", URL]) == 0

    out = capsys.readouterr().out
    assert "webhook registered" in out
    assert "token-value" not in out
    assert "w" * 24 not in out
    assert bot.closed
    assert bot.calls[0][1]["url"] == URL


def test_main_reports_failures_without_the_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _settings(monkeypatch)
    bot = FakeBot(fail=RuntimeError("https://api.telegram.org/bot123456:token-value/setWebhook"))
    monkeypatch.setattr(polling, "create_bot", lambda *_args: bot)

    assert polling.main(["--set-webhook", URL]) == 1

    captured = capsys.readouterr()
    assert "token-value" not in captured.out + captured.err
    assert bot.closed


def test_main_delete_webhook_passes_the_drop_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    bot = FakeBot()
    monkeypatch.setattr(polling, "create_bot", lambda *_args: bot)

    assert polling.main(["--delete-webhook", "--drop-pending"]) == 0

    assert ("delete", {"drop_pending_updates": True}) in bot.calls


def test_main_webhook_info(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch)
    bot = FakeBot()
    monkeypatch.setattr(polling, "create_bot", lambda *_args: bot)

    assert polling.main(["--webhook-info"]) == 0
    assert [name for name, _ in bot.calls] == ["info"]


def test_main_rejects_bad_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch)
    for argv in (
        ["--set-webhook", "http://insecure.example/x"],
        ["--poll", "--set-webhook", URL],
        ["--drop-pending"],
        [],
    ):
        with pytest.raises(SystemExit) as excinfo:
            polling.main(argv)
        assert excinfo.value.code == 2


def test_main_set_webhook_needs_token_and_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    get_settings.cache_clear()
    with pytest.raises(SystemExit) as excinfo:
        polling.main(["--set-webhook", URL])
    assert excinfo.value.code == 2

    _settings(monkeypatch, secret="")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")
    get_settings.cache_clear()
    with pytest.raises(SystemExit) as excinfo:
        polling.main(["--set-webhook", URL])
    assert excinfo.value.code == 2
