"""Tests for environment-driven settings."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from scambot.core.settings import (
    DEV_USER_ID_PEPPER,
    ENV_FILE,
    PROJECT_ROOT,
    Settings,
    get_settings,
)


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


PROD_ENV = {
    "ENVIRONMENT": "prod",
    "TELEGRAM_BOT_TOKEN": "123456:bot-token-value",
    "TELEGRAM_WEBHOOK_SECRET": "w" * 32,
    "USER_ID_PEPPER": "a-unique-production-secret",
    "DATABASE_URL": "postgresql+asyncpg://scambot:s3cret-pw@postgres:5432/scambot",
    "REDIS_URL": "redis://redis:6379/0",
}


@pytest.fixture
def prod_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name, value in PROD_ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_a_complete_production_environment_is_accepted(prod_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).environment == "prod"


def test_production_requires_a_real_pepper(prod_env: pytest.MonkeyPatch) -> None:
    prod_env.delenv("USER_ID_PEPPER")

    with pytest.raises(ValidationError, match="USER_ID_PEPPER") as excinfo:
        Settings(_env_file=None)

    assert "TELEGRAM" not in str(excinfo.value)


def test_production_rejects_a_blank_or_short_pepper(prod_env: pytest.MonkeyPatch) -> None:
    for pepper in ("   ", "short"):
        prod_env.setenv("USER_ID_PEPPER", pepper)
        with pytest.raises(ValidationError, match="USER_ID_PEPPER"):
            Settings(_env_file=None)


@pytest.mark.parametrize(
    "name", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET", "DATABASE_URL", "REDIS_URL"]
)
def test_production_names_each_missing_setting(prod_env: pytest.MonkeyPatch, name: str) -> None:
    prod_env.delenv(name)

    with pytest.raises(ValidationError, match=name):
        Settings(_env_file=None)


@pytest.mark.parametrize("secret", ["short", "has space " * 4, "bad/char" * 4, "x" * 257])
def test_production_checks_the_webhook_secret_format(
    prod_env: pytest.MonkeyPatch, secret: str
) -> None:
    prod_env.setenv("TELEGRAM_WEBHOOK_SECRET", secret)

    with pytest.raises(ValidationError, match="TELEGRAM_WEBHOOK_SECRET"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://scambot:scambot@postgres:5432/scambot",
        "postgresql+asyncpg://scambot:good-password@localhost:5432/scambot",
        "postgresql+asyncpg://scambot@postgres:5432/scambot",
        "not a url",
    ],
)
def test_production_rejects_default_or_local_database_settings(
    prod_env: pytest.MonkeyPatch, url: str
) -> None:
    prod_env.setenv("DATABASE_URL", url)

    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(_env_file=None)


def test_production_rejects_a_local_redis(prod_env: pytest.MonkeyPatch) -> None:
    prod_env.setenv("REDIS_URL", "redis://localhost:6379/0")

    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(_env_file=None)


def test_development_keeps_the_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "dev")

    settings = Settings(_env_file=None)

    assert "localhost" in settings.database_url.get_secret_value()


def test_production_whatsapp_needs_a_verify_token(prod_env: pytest.MonkeyPatch) -> None:
    prod_env.setenv("WHATSAPP_TOKEN", "wa-token")
    prod_env.setenv("WHATSAPP_PHONE_NUMBER_ID", "123")
    prod_env.setenv("WHATSAPP_APP_SECRET", "app-secret")

    with pytest.raises(ValidationError, match="WHATSAPP_VERIFY_TOKEN"):
        Settings(_env_file=None)

    prod_env.setenv("WHATSAPP_VERIFY_TOKEN", "verify")
    assert Settings(_env_file=None).whatsapp_enabled


def test_production_admin_hash_must_be_argon2_when_set(prod_env: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).admin_password_hash.get_secret_value() == ""
    prod_env.setenv("ADMIN_PASSWORD_HASH", "argon2id$v=19$mangled")

    with pytest.raises(ValidationError, match="ADMIN_PASSWORD_HASH"):
        Settings(_env_file=None)

    prod_env.setenv("ADMIN_PASSWORD_HASH", "$argon2id$v=19$m=1,t=1,p=1$abc$def")
    assert Settings(_env_file=None).admin_password_hash.get_secret_value()


def test_production_reports_every_problem_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "bad secret value!")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:hunter2@localhost/db")

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    text = str(excinfo.value)
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET", "USER_ID_PEPPER", "DATABASE_URL"):
        assert name in text
    assert "hunter2" not in text
    assert "bad secret value!" not in text


def test_validation_errors_do_not_echo_secret_input(prod_env: pytest.MonkeyPatch) -> None:
    prod_env.delenv("USER_ID_PEPPER")
    prod_env.setenv("GOOGLE_API_KEY", "google-key-that-must-stay-hidden")

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


def test_llm_and_checker_server_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_model_analyze == "claude-opus-5-5"
    assert settings.llm_effort_scam_classify == "low"
    assert settings.llm_max_tokens_explanation == 4096
    assert settings.llm_refusal_fallback is True
    assert settings.mcp_url == "http://localhost:8001/mcp"
    assert settings.mcp_allowed_hosts == ("localhost:*", "127.0.0.1:*")
    assert settings.expand_short_urls is True
    assert settings.llm_model_factcheck_relevance == "claude-opus-5-5"
    assert settings.llm_effort_factcheck_relevance == "low"
    assert settings.llm_max_tokens_factcheck_relevance == 8192
    assert settings.llm_model_caption_compare == "claude-opus-5-5"
    assert settings.llm_effort_caption_compare == "low"
    assert settings.llm_max_tokens_caption_compare == 8192
    assert settings.embedding_model == "intfloat/multilingual-e5-small"


def test_llm_and_checker_server_settings_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp:*, localhost:*")
    monkeypatch.setenv("LLM_EFFORT_EXPLANATION", "high")
    monkeypatch.setenv("LLM_REFUSAL_FALLBACK", "false")
    monkeypatch.setenv("MCP_PORT", "9001")
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("LLM_MAX_TOKENS_FACTCHECK_RELEVANCE", "2048")

    settings = Settings(_env_file=None)

    assert settings.mcp_allowed_hosts == ("mcp:*", "localhost:*")
    assert settings.llm_effort_explanation == "high"
    assert settings.llm_refusal_fallback is False
    assert settings.mcp_port == 9001
    assert settings.embedding_model == "intfloat/multilingual-e5-base"
    assert settings.llm_max_tokens_factcheck_relevance == 2048


def test_unknown_effort_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EFFORT_ANALYZE", "extreme")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_example_lists_every_setting() -> None:
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    names = {line.split("=", 1)[0] for line in example.splitlines() if "=" in line}
    assert {name.upper() for name in Settings.model_fields} <= names


def test_media_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.media_dir.name == "scambot-media"
    assert settings.ocr_engine == "tesseract"
    assert settings.stt_engine == "faster_whisper"
    assert (settings.whisper_model, settings.whisper_compute_type) == ("small", "int8")
    assert settings.whisper_cpu_threads == 0


def test_robustness_overrides_default_to_none_and_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = Settings(_env_file=None)
    assert (defaults.rate_limit_per_hour, defaults.cache_window_days) == (None, None)

    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "5")
    monkeypatch.setenv("CACHE_WINDOW_DAYS", "0.5")
    overridden = Settings(_env_file=None)

    assert (overridden.rate_limit_per_hour, overridden.cache_window_days) == (5, 0.5)


@pytest.mark.parametrize("name", ["RATE_LIMIT_PER_HOUR", "CACHE_WINDOW_DAYS"])
def test_robustness_overrides_must_be_positive(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_admin_chat_id_defaults_to_off() -> None:
    assert Settings(_env_file=None).admin_chat_id == ""
    assert Settings(_env_file=None, admin_chat_id="123").admin_chat_id == "123"


def test_whatsapp_defaults_and_enabling() -> None:
    off = Settings(_env_file=None)

    assert not off.whatsapp_enabled
    assert (off.whatsapp_api_version, off.whatsapp_mark_read) == ("v23.0", True)
    assert off.whatsapp_graph_url == "https://graph.facebook.com"
    on = Settings(
        _env_file=None,
        whatsapp_token=SecretStr("t"),
        whatsapp_app_secret=SecretStr("s"),
        whatsapp_phone_number_id="123",
        whatsapp_allowlist="+91 98000 00001, 919800000002",  # type: ignore[arg-type]
    )
    assert on.whatsapp_enabled
    assert on.whatsapp_allowlist == ("+91 98000 00001", "919800000002")


def test_whatsapp_needs_an_app_secret_and_never_echoes_secrets() -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(
            _env_file=None, whatsapp_token=SecretStr("tok-123"), whatsapp_phone_number_id="123"
        )

    assert "WHATSAPP_APP_SECRET" in str(caught.value)
    assert "tok-123" not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("whatsapp_api_version", "23"),
        ("whatsapp_api_version", "v23"),
        ("whatsapp_graph_url", "http://graph.facebook.com"),
        ("whatsapp_graph_url", "https://"),
        ("whatsapp_phone_number_id", "12ab"),
    ],
)
def test_whatsapp_values_are_validated(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: value})


@pytest.mark.parametrize(
    ("name", "url"),
    [
        ("DATABASE_URL", "postgresql+asyncpg://scambot:p@ss/word@postgres:5432/scambot"),
        ("DATABASE_URL", "postgresql+asyncpg://scambot:pa#ss@postgres:5432/scambot"),
        ("DATABASE_URL", "postgresql+asyncpg://scambot:pa$word!@postgres:5432/scambot"),
        ("REDIS_URL", "redis://:pa/ss@redis:6379/0"),
        ("REDIS_URL", "redis://:pa#ss@redis:6379/0"),
        ("REDIS_URL", "redis://:pa ss@redis:6379/0"),
    ],
)
def test_production_rejects_passwords_that_are_not_url_safe(
    prod_env: pytest.MonkeyPatch, name: str, url: str
) -> None:
    prod_env.setenv(name, url)

    with pytest.raises(ValidationError, match=name):
        Settings(_env_file=None)


def test_production_accepts_hex_and_percent_encoded_passwords(
    prod_env: pytest.MonkeyPatch,
) -> None:
    prod_env.setenv("DATABASE_URL", f"postgresql+asyncpg://scambot:{'ab12' * 12}@postgres/scambot")
    prod_env.setenv("REDIS_URL", "redis://:p%40ss-w.o_r~d@redis:6379/0")

    Settings(_env_file=None)
