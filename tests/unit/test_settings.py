"""Tests for environment-driven settings."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from scambot.core.settings import DEV_USER_ID_PEPPER, ENV_FILE, Settings, get_settings


def test_defaults_are_safe_for_local_development() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "dev"
    assert settings.log_level == "INFO"
    assert settings.telegram_bot_token.get_secret_value() == ""
    assert settings.telegram_api_url == "https://api.telegram.org"
    assert settings.whatsapp_allowlist == ()
    assert settings.user_id_pepper.get_secret_value() == DEV_USER_ID_PEPPER
    assert settings.database_url.get_secret_value().startswith("postgresql+asyncpg://")
    assert settings.redis_url.get_secret_value().startswith("redis://")


def test_reads_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:abc")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("SIGHTENGINE_USER", "user-1")

    settings = Settings(_env_file=None)

    assert settings.telegram_bot_token.get_secret_value() == "123456:abc"
    assert settings.log_level == "DEBUG"
    assert settings.sightengine_user == "user-1"


@pytest.mark.parametrize("url", ["api.telegram.org", "ftp://api.telegram.org", "https://"])
def test_telegram_api_url_must_be_http(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("TELEGRAM_API_URL", url)

    with pytest.raises(ValidationError, match="TELEGRAM_API_URL"):
        Settings(_env_file=None)


def test_reads_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-file\nUNKNOWN_VARIABLE=ignored\n")

    settings = Settings(_env_file=env_file)

    assert settings.anthropic_api_key.get_secret_value() == "from-file"


def test_environment_overrides_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=from-file\n")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")

    settings = Settings(_env_file=env_file)

    assert settings.google_api_key.get_secret_value() == "from-env"


def test_empty_values_fall_back_to_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("USER_ID_PEPPER=\nLOG_LEVEL=\n")
    monkeypatch.setenv("ADMIN_USER", "")

    settings = Settings(_env_file=env_file)

    assert settings.user_id_pepper.get_secret_value() == DEV_USER_ID_PEPPER
    assert settings.log_level == "INFO"
    assert settings.admin_user == "admin"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("919800000001", ("919800000001",)),
        (" 919800000001 , 919800000002,, ", ("919800000001", "919800000002")),
    ],
)
def test_whatsapp_allowlist_is_comma_separated(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: tuple[str, ...]
) -> None:
    monkeypatch.setenv("WHATSAPP_ALLOWLIST", raw)

    assert Settings(_env_file=None).whatsapp_allowlist == expected


def test_secrets_are_masked_in_repr_and_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret-value")

    settings = Settings(_env_file=None)

    assert "sk-ant-very-secret-value" not in repr(settings)
    assert "sk-ant-very-secret-value" not in settings.model_dump_json()
    assert isinstance(settings.anthropic_api_key, SecretStr)


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_production_requires_a_real_pepper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")

    with pytest.raises(ValidationError, match="USER_ID_PEPPER"):
        Settings(_env_file=None)

    monkeypatch.setenv("USER_ID_PEPPER", "a-unique-production-secret")
    assert Settings(_env_file=None).environment == "prod"


def test_production_rejects_a_blank_pepper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("USER_ID_PEPPER", "   ")

    with pytest.raises(ValidationError, match="USER_ID_PEPPER"):
        Settings(_env_file=None)


def test_validation_errors_do_not_echo_secret_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key-that-must-stay-hidden")

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    assert "google-key-that-must-stay-hidden" not in str(excinfo.value)


def test_dotenv_path_does_not_depend_on_the_working_directory() -> None:
    assert ENV_FILE.is_absolute()
    assert ENV_FILE.name == ".env"
    assert (ENV_FILE.parent / "pyproject.toml").is_file()


def test_settings_are_immutable() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.admin_user = "someone-else"  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
