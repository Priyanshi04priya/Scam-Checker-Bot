"""Short-lived cache of checker results, keyed by tool name and the tool's input."""

from collections.abc import Mapping

from arq.connections import ArqRedis

from scambot.core.contracts import CheckerName, CheckerResult
from scambot.core.logging import get_logger
from scambot.core.redis import checker_key

_log = get_logger(component="checker_cache")


class CheckerCache:
    """Only results that carry information are kept; an error is never stored or served."""

    def __init__(
        self, redis: ArqRedis, ttls: Mapping[CheckerName, int], inconclusive_ttl_s: int
    ) -> None:
        self._redis = redis
        self._ttls = ttls
        self._inconclusive_ttl_s = inconclusive_ttl_s

    async def get(self, checker: CheckerName, key: str) -> CheckerResult | None:
        try:
            raw = await self._redis.get(checker_key(checker, key))
            if raw is None:
                return None
            result = CheckerResult.model_validate_json(raw)
        except Exception as exc:  # a broken entry or an unreachable Redis is a miss
            _log.warning("checker_cache_read_failed", checker=checker, error=type(exc).__name__)
            return None
        return result if result.status != "error" and result.checker == checker else None

    async def put(self, checker: CheckerName, key: str, result: CheckerResult) -> None:
        ttl = self._ttls.get(checker)
        if ttl is None or result.status not in ("ok", "inconclusive"):
            return
        if result.status == "inconclusive":
            ttl = min(ttl, self._inconclusive_ttl_s)
        try:
            await self._redis.set(checker_key(checker, key), result.model_dump_json(), ex=ttl)
        except Exception as exc:
            _log.warning("checker_cache_write_failed", checker=checker, error=type(exc).__name__)
