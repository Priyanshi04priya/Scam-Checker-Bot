"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
import structlog

from scambot.core.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every settings variable from the environment, stop the project ``.env`` from
    being read and clear the settings cache."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """Restore structlog's default configuration after each test."""
    yield
    structlog.reset_defaults()
