"""Application settings loaded from environment variables and an optional ``.env`` file."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEV_USER_ID_PEPPER = "dev-only-pepper-do-not-use-in-production"

# The project root's .env, so the same file is read whatever the working directory is.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["dev", "test", "prod"]


class Settings(BaseSettings):
    """Every environment variable the service reads, with safe local-development defaults."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        # Validation errors would otherwise echo the raw (unmasked) input, secrets included.
        hide_input_in_errors=True,
    )

    environment: Environment = "dev"
    log_level: LogLevel = "INFO"

    telegram_bot_token: SecretStr = SecretStr("")
    telegram_webhook_secret: SecretStr = SecretStr("")

    whatsapp_token: SecretStr = SecretStr("")
    whatsapp_app_secret: SecretStr = SecretStr("")
    whatsapp_allowlist: Annotated[tuple[str, ...], NoDecode] = ()

    anthropic_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    sightengine_user: str = ""
    sightengine_secret: SecretStr = SecretStr("")

    user_id_pepper: SecretStr = SecretStr(DEV_USER_ID_PEPPER)

    admin_user: str = "admin"
    admin_password_hash: SecretStr = SecretStr("")

    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://scambot:scambot@localhost:5432/scambot"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("whatsapp_allowlist", mode="before")
    @classmethod
    def _split_allowlist(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def _reject_dev_pepper_in_prod(self) -> Self:
        if self.environment == "prod" and (
            self.user_id_pepper.get_secret_value().strip() in {"", DEV_USER_ID_PEPPER}
        ):
            msg = "USER_ID_PEPPER must be set to a unique secret value in production"
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached after the first call)."""
    return Settings()
