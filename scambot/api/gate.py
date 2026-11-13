"""Admission control for new checks: blocked users and the hourly limit."""

import datetime as dt
import enum
import math
from collections.abc import Callable
from dataclasses import dataclass

from arq.connections import ArqRedis

from scambot.core.logging import get_logger
from scambot.core.redis import hit_rate, refund_rate
from scambot.core.runtime import RuntimeConfig

_log = get_logger(component="gate")


class Admission(enum.Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    LIMITED = "limited"


@dataclass(frozen=True)
class Decision:
    admission: Admission
    limit: int = 0
    minutes: int = 0


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Gate:
    """Decides whether a user's next check is accepted. It never fails the update itself."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        redis: ArqRedis,
        window_s: int,
        clock: Callable[[], dt.datetime] = _utc_now,
    ) -> None:
        self._runtime = runtime
        self._redis = redis
        self._window_s = window_s
        self._clock = clock

    async def admit(self, user_hash: str) -> Decision:
        if await self._runtime.is_blocked(user_hash):
            _log.info("user_blocked", user_hash=user_hash)
            return Decision(Admission.BLOCKED)
        limit = await self._runtime.rate_limit()
        try:
            count, ttl = await hit_rate(self._redis, user_hash, self._clock(), self._window_s)
        except Exception as exc:  # without the counter the check is better allowed than lost
            _log.warning("rate_limit_failed", error=type(exc).__name__)
            return Decision(Admission.ALLOWED)
        if count > limit:
            _log.info("rate_limited", user_hash=user_hash, count=count, limit=limit)
            return Decision(Admission.LIMITED, limit=limit, minutes=max(1, math.ceil(ttl / 60)))
        return Decision(Admission.ALLOWED)

    async def refund(self, user_hash: str) -> None:
        """Undo the charge of an admitted check that was not queued; best effort."""
        try:
            await refund_rate(self._redis, user_hash, self._clock())
        except Exception as exc:
            _log.warning("rate_refund_failed", error=type(exc).__name__)
