"""What the admin pages need, built once per process and shared with the intake."""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

from arq.connections import ArqRedis

from scambot.api.admin.queries import StatsStore
from scambot.core.config import AppConfig
from scambot.core.runtime import RuntimeConfig, SettingsStore


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class AdminDeps:
    stats: StatsStore
    settings_store: SettingsStore
    runtime: RuntimeConfig
    redis: ArqRedis
    config: AppConfig
    now: Callable[[], dt.datetime] = field(default=utc_now)
