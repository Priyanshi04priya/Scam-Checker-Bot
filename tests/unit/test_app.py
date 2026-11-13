"""Tests for the API startup wiring and the polling entry point."""

from typing import Any

import pytest
from arq.connections import ArqRedis
from pydantic import SecretStr

from scambot.api import intake as intake_module
from scambot.api import main as main_module
from scambot.api import polling
from scambot.api.intake import Intake
from scambot.api.main import create_app
from scambot.core.settings import Settings
from tests.whatsapp import settings as whatsapp_settings

TOKEN = "123456:" + "A" * 35


async def test_lifespan_opens_and_closes_the_telegram_intake(
    redis: ArqRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_create_redis(_url: SecretStr) -> ArqRedis:
        return redis

    monkeypatch.setattr(intake_module, "create_redis", fake_create_redis)
    monkeypatch.setattr(main_module, "create_redis", fake_create_redis)
    app = create_app(Settings(_env_file=None, telegram_bot_token=SecretStr(TOKEN)))

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.telegram_intake, Intake)


async def test_lifespan_opens_the_whatsapp_intake_only_when_enabled(
    redis: ArqRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_create_redis(_url: SecretStr) -> ArqRedis:
        return redis

    monkeypatch.setattr(intake_module, "create_redis", fake_create_redis)
    monkeypatch.setattr(main_module, "create_redis", fake_create_redis)

    enabled = create_app(whatsapp_settings(telegram_bot_token=SecretStr(TOKEN)))
    async with enabled.router.lifespan_context(enabled):
        assert isinstance(enabled.state.whatsapp_intake, Intake)

    disabled = create_app(Settings(_env_file=None, telegram_bot_token=SecretStr(TOKEN)))
    async with disabled.router.lifespan_context(disabled):
        assert disabled.state.whatsapp_intake is None


def test_polling_requires_the_poll_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        polling.main([])

    assert exc.value.code == 2
    assert "--poll" in capsys.readouterr().err


def test_polling_requires_a_token(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        polling.main(["--poll"])

    assert exc.value.code == 2
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_polling_runs_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    calls: list[Any] = []

    async def fake_run_polling() -> None:
        calls.append("ran")

    monkeypatch.setattr(polling, "run_polling", fake_run_polling)

    assert polling.main(["--poll"]) == 0
    assert calls == ["ran"]


async def test_run_polling_wires_intake_into_the_loop(
    redis: ArqRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)

    async def fake_create_redis(_url: SecretStr) -> ArqRedis:
        return redis

    handlers: list[Any] = []

    async def fake_poll_updates(_bot: object, handler: Any) -> None:
        handlers.append(handler)

    monkeypatch.setattr(intake_module, "create_redis", fake_create_redis)
    monkeypatch.setattr(polling, "poll_updates", fake_poll_updates)

    await polling.run_polling()

    (handler,) = handlers
    assert isinstance(handler.__self__, Intake)
