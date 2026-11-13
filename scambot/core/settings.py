"""Application settings loaded from environment variables and an optional ``.env`` file."""

import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, StringConstraints, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import make_url

DEV_USER_ID_PEPPER = "dev-only-pepper-do-not-use-in-production"

# Paths are fixed relative to the project root, so they work whatever the working directory is.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_DIR = PROJECT_ROOT / "config"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["dev", "test", "prod"]
_WEBHOOK_SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,256}$")
_LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1"})
_URL_SAFE_PASSWORD = re.compile(r"(?:[A-Za-z0-9._~-]|%[0-9A-Fa-f]{2})*")
_DEFAULT_DB_PASSWORDS = frozenset({"", "scambot", "postgres", "password"})
Effort = Literal["low", "medium", "high", "xhigh", "max"]


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
    telegram_api_url: str = "https://api.telegram.org"

    whatsapp_token: SecretStr = SecretStr("")
    whatsapp_app_secret: SecretStr = SecretStr("")
    whatsapp_verify_token: SecretStr = SecretStr("")
    whatsapp_phone_number_id: Annotated[str, StringConstraints(pattern=r"^\d*$")] = ""
    whatsapp_api_version: Annotated[str, StringConstraints(pattern=r"^v\d+\.\d+$")] = "v23.0"
    whatsapp_graph_url: str = "https://graph.facebook.com"
    whatsapp_mark_read: bool = True
    whatsapp_allowlist: Annotated[tuple[str, ...], NoDecode] = ()

    anthropic_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    sightengine_user: str = ""
    sightengine_secret: SecretStr = SecretStr("")

    user_id_pepper: SecretStr = SecretStr(DEV_USER_ID_PEPPER)

    admin_user: str = "admin"
    admin_password_hash: SecretStr = SecretStr("")
    admin_chat_id: str = ""

    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://scambot:scambot@localhost:5432/scambot"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # Model, effort and output-token budget per LLM call (thinking counts toward the budget).
    llm_model_analyze: str = "claude-opus-5-5"
    llm_model_scam_classify: str = "claude-opus-5-5"
    llm_model_explanation: str = "claude-opus-5-5"
    llm_model_factcheck_relevance: str = "claude-opus-5-5"
    llm_model_caption_compare: str = "claude-opus-5-5"
    llm_effort_analyze: Effort = "low"
    llm_effort_scam_classify: Effort = "low"
    llm_effort_explanation: Effort = "low"
    llm_effort_factcheck_relevance: Effort = "low"
    llm_effort_caption_compare: Effort = "low"
    llm_max_tokens_analyze: int = Field(default=8192, gt=0)
    llm_max_tokens_scam_classify: int = Field(default=8192, gt=0)
    llm_max_tokens_explanation: int = Field(default=4096, gt=0)
    llm_max_tokens_factcheck_relevance: int = Field(default=8192, gt=0)
    llm_max_tokens_caption_compare: int = Field(default=8192, gt=0)
    llm_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0)
    llm_refusal_fallback: bool = True

    mcp_url: str = "http://localhost:8001/mcp"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8001, gt=0, lt=65536)
    mcp_allowed_hosts: Annotated[tuple[str, ...], NoDecode] = ("localhost:*", "127.0.0.1:*")

    expand_short_urls: bool = True

    # Sentence-embedding model for the fact-check index (needs the optional embeddings group).
    embedding_model: str = "intfloat/multilingual-e5-small"

    # Media: scratch directory for downloaded files, and the text-extraction engines.
    media_dir: Path = Path(tempfile.gettempdir()) / "scambot-media"
    ocr_engine: Literal["tesseract", "none"] = "tesseract"
    stt_engine: Literal["faster_whisper", "none"] = "faster_whisper"
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = Field(default=0, ge=0)

    # Synthetic-voice detection. "huggingface" runs a local model (needs the optional voice
    # group); "reality_defender" is a hosted service; "none" switches the voice check off.
    voice_backend: Literal["huggingface", "reality_defender", "none"] = "huggingface"
    voice_model_id: str = "MelodyMachine/Deepfake-audio-detection-V2"
    voice_model_revision: str | None = None
    voice_cpu_threads: int = Field(default=0, ge=0)
    reality_defender_api_key: SecretStr = SecretStr("")

    # Optional overrides of thresholds.yaml; the settings table in the database wins over both.
    rate_limit_per_hour: int | None = Field(default=None, gt=0)
    cache_window_days: float | None = Field(default=None, gt=0)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("telegram_api_url")
    @classmethod
    def _check_telegram_api_url(cls, value: str) -> str:
        # A malformed base URL makes every Bot API error message echo the request URL, which
        # contains the bot token.
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            msg = "TELEGRAM_API_URL must be an http:// or https:// URL"
            raise ValueError(msg)
        return value

    @field_validator("whatsapp_graph_url")
    @classmethod
    def _check_whatsapp_graph_url(cls, value: str) -> str:
        # The access token is sent to this host, so only https is accepted.
        parts = urlsplit(value)
        if parts.scheme != "https" or not parts.hostname:
            msg = "WHATSAPP_GRAPH_URL must be an https:// URL"
            raise ValueError(msg)
        return value.rstrip("/")

    @field_validator("whatsapp_allowlist", "mcp_allowed_hosts", mode="before")
    @classmethod
    def _split_allowlist(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @property
    def whatsapp_enabled(self) -> bool:
        """WhatsApp is served when both the access token and the phone number ID are set."""
        return bool(self.whatsapp_token.get_secret_value() and self.whatsapp_phone_number_id)

    @model_validator(mode="after")
    def _require_whatsapp_app_secret(self) -> Self:
        if self.whatsapp_enabled and not self.whatsapp_app_secret.get_secret_value():
            msg = "WHATSAPP_APP_SECRET must be set when WhatsApp is enabled"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_production_secrets(self) -> Self:
        if self.environment != "prod":
            return self
        problems = [*self._prod_secret_problems(), *self._prod_url_problems()]
        if problems:
            msg = "invalid production configuration: " + "; ".join(problems)
            raise ValueError(msg)
        return self

    def _prod_secret_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.telegram_bot_token.get_secret_value():
            problems.append("TELEGRAM_BOT_TOKEN must be set")
        if not _WEBHOOK_SECRET_PATTERN.fullmatch(self.telegram_webhook_secret.get_secret_value()):
            problems.append(
                "TELEGRAM_WEBHOOK_SECRET must be 16-256 characters from A-Z, a-z, 0-9, _ and -"
            )
        pepper = self.user_id_pepper.get_secret_value().strip()
        if pepper in {"", DEV_USER_ID_PEPPER}:
            problems.append("USER_ID_PEPPER must be set to a unique secret value in production")
        elif len(pepper) < 16:
            problems.append("USER_ID_PEPPER must be at least 16 characters")
        if self.whatsapp_enabled and not self.whatsapp_verify_token.get_secret_value():
            problems.append("WHATSAPP_VERIFY_TOKEN must be set when WhatsApp is enabled")
        admin_hash = self.admin_password_hash.get_secret_value()
        if admin_hash and not admin_hash.startswith("$argon2"):
            problems.append(
                "ADMIN_PASSWORD_HASH is not an argon2 hash (wrap it in single quotes in .env, "
                "and write every $ as $$ inside compose files)"
            )
        return problems

    def _prod_url_problems(self) -> list[str]:
        problems: list[str] = []
        try:
            db = make_url(self.database_url.get_secret_value())
        except Exception:
            problems.append("DATABASE_URL is not a valid URL")
        else:
            if (db.host or "") in _LOCAL_HOSTS:
                problems.append("DATABASE_URL must not point at localhost in production")
            if (db.password or "") in _DEFAULT_DB_PASSWORDS:
                problems.append("DATABASE_URL must use a non-default database password")
        try:
            redis_host = urlsplit(self.redis_url.get_secret_value()).hostname or ""
        except ValueError:
            redis_host = ""
        if redis_host in _LOCAL_HOSTS:
            problems.append("REDIS_URL must not point at localhost in production")
        problems.extend(
            f"{name} must have a URL-safe password (generate one with: openssl rand -hex 24)"
            for name, url in (
                ("DATABASE_URL", self.database_url.get_secret_value()),
                ("REDIS_URL", self.redis_url.get_secret_value()),
            )
            if not _url_credentials_are_safe(url)
        )
        return problems


def _url_credentials_are_safe(url: str) -> bool:
    """Whether the password in ``url`` survives URL parsing (hex or percent-encoded only)."""
    try:
        parts = urlsplit(url)
        password = parts.password
        host = parts.hostname
    except ValueError:
        return False
    if "@" in url and (not host or "@" in f"{parts.path}{parts.query}{parts.fragment}"):
        return False
    return password is None or _URL_SAFE_PASSWORD.fullmatch(password) is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached after the first call)."""
    return Settings()
