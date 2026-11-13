"""Settings an operator can change while the service runs.

Order of precedence: the ``settings`` table, then the environment, then ``thresholds.yaml``.
Values are cached in Redis for a short time. Any failure falls back to the defaults, so the
database or Redis being down never stops a message from being handled.
"""

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Final, Protocol

from arq.connections import ArqRedis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.config import FactcheckSource, SourcesConfig, ThresholdsConfig
from scambot.core.logging import get_logger
from scambot.core.models import Setting, User
from scambot.core.redis import config_key
from scambot.core.settings import Settings

_log = get_logger(component="runtime")

RATE_LIMIT_KEY: Final = "rate_limit_per_hour"
CACHE_WINDOW_KEY: Final = "cache_window_days"
SOURCES_KEY: Final = "factcheck_sources"
MAX_SOURCES: Final = 50
_NONE: Final = "null"


class SettingsStore(Protocol):
    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any) -> None:
        """Insert or replace a setting."""
        ...

    async def delete(self, key: str) -> None:
        """Remove a setting so the default applies again."""
        ...

    async def is_blocked(self, user_hash: str) -> bool: ...

    async def set_blocked(self, user_hash: str, blocked: bool) -> None:
        """Block or unblock a user, creating the user row if needed."""
        ...


class SqlSettingsStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, key: str) -> Any | None:
        async with self._sessions() as session:
            row = await session.get(Setting, key)
            return None if row is None else row.value

    async def set(self, key: str, value: Any) -> None:
        async with self._sessions() as session, session.begin():
            row = await session.get(Setting, key)
            if row is None:
                session.add(Setting(key=key, value=value))
            else:
                row.value = value

    async def delete(self, key: str) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(Setting).where(Setting.key == key))

    async def is_blocked(self, user_hash: str) -> bool:
        async with self._sessions() as session:
            result = await session.execute(select(User.blocked).where(User.user_hash == user_hash))
            return bool(result.scalar_one_or_none())

    async def set_blocked(self, user_hash: str, blocked: bool) -> None:
        async with self._sessions() as session, session.begin():
            user = await session.get(User, user_hash)
            if user is None:
                session.add(User(user_hash=user_hash, blocked=blocked))
            else:
                user.blocked = blocked


@dataclass(frozen=True)
class RuntimeDefaults:
    rate_per_hour: int
    window_days: float
    rate_origin: str = "file"
    window_origin: str = "file"

    @classmethod
    def build(cls, thresholds: ThresholdsConfig, settings: Settings) -> "RuntimeDefaults":
        """The environment wins over the YAML values."""
        return cls(
            rate_per_hour=settings.rate_limit_per_hour or thresholds.limits.rate_per_hour,
            window_days=settings.cache_window_days or thresholds.cache.verdict_window_days,
            rate_origin="env" if settings.rate_limit_per_hour else "file",
            window_origin="env" if settings.cache_window_days else "file",
        )


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return None
    return float(value)


class RuntimeConfig:
    def __init__(
        self,
        store: SettingsStore,
        redis: ArqRedis,
        defaults: RuntimeDefaults,
        cache_s: int,
    ) -> None:
        self._store = store
        self._redis = redis
        self._defaults = defaults
        self._cache_s = cache_s

    async def _setting(self, name: str) -> float | None:
        """The override for ``name`` if one is set and valid; None means use the default."""
        key = config_key(name)
        cached = await self._redis.get(key)
        if cached is not None:
            return _positive(json.loads(cached))
        stored = await self._store.get(name)
        valid = _positive(stored)
        if stored is not None and valid is None:
            _log.warning("setting_invalid", name=name)
        await self._redis.set(key, _NONE if valid is None else json.dumps(valid), ex=self._cache_s)
        return valid

    async def rate_limit(self) -> int:
        try:
            override = await self._setting(RATE_LIMIT_KEY)
        except Exception as exc:  # the default is safe; never fail the update over it
            _log.warning("setting_failed", name=RATE_LIMIT_KEY, error=type(exc).__name__)
            override = None
        return max(1, int(override)) if override else self._defaults.rate_per_hour

    async def verdict_window(self) -> dt.timedelta:
        try:
            override = await self._setting(CACHE_WINDOW_KEY)
        except Exception as exc:
            _log.warning("setting_failed", name=CACHE_WINDOW_KEY, error=type(exc).__name__)
            override = None
        return dt.timedelta(days=override or self._defaults.window_days)

    async def is_blocked(self, user_hash: str) -> bool:
        key = config_key(f"blocked:{user_hash}")
        try:
            cached = await self._redis.get(key)
            if cached is not None:
                return bool(json.loads(cached))
            blocked = await self._store.is_blocked(user_hash)
            await self._redis.set(key, json.dumps(blocked), ex=self._cache_s)
        except Exception as exc:
            _log.warning("blocked_check_failed", error=type(exc).__name__)
            return False
        return blocked

    async def describe(self) -> dict[str, dict[str, Any]]:
        """Each limit with its effective value and where it comes from.

        Reads the database directly, so the answer never lags behind the cached values.
        """
        defaults = self._defaults
        rate = _positive(await self._store.get(RATE_LIMIT_KEY))
        window = _positive(await self._store.get(CACHE_WINDOW_KEY))
        return {
            RATE_LIMIT_KEY: {
                "value": max(1, int(rate)) if rate else defaults.rate_per_hour,
                "origin": "table" if rate else defaults.rate_origin,
            },
            CACHE_WINDOW_KEY: {
                "value": window or defaults.window_days,
                "origin": "table" if window else defaults.window_origin,
            },
        }

    async def invalidate(self, name: str) -> None:
        """Forget the cached value so the next read goes to the database."""
        await self._redis.delete(config_key(name))


def merged_sources(baseline: SourcesConfig, sources: list[FactcheckSource]) -> SourcesConfig:
    """``baseline`` with its publisher list replaced; raises ValueError when the list is invalid."""
    if not 0 < len(sources) <= MAX_SOURCES:
        msg = f"the list needs between 1 and {MAX_SOURCES} publishers"
        raise ValueError(msg)
    return SourcesConfig(sources=sources, credible=baseline.credible, ratings=baseline.ratings)


async def effective_sources(store: SettingsStore, baseline: SourcesConfig) -> SourcesConfig:
    """The publishers to index: the list saved from the admin page, else the file's list.

    Only the publisher list can be replaced; ratings and credible sites stay in the file. A
    saved list that no longer validates is ignored.
    """
    try:
        stored = await store.get(SOURCES_KEY)
    except Exception as exc:  # the file's list is always a safe answer
        _log.warning("setting_failed", name=SOURCES_KEY, error=type(exc).__name__)
        return baseline
    if stored is None:
        return baseline
    try:
        if not isinstance(stored, list):
            msg = "not a list"
            raise ValueError(msg)
        return merged_sources(baseline, [FactcheckSource.model_validate(item) for item in stored])
    except ValueError:
        _log.warning("setting_invalid", name=SOURCES_KEY)
        return baseline
