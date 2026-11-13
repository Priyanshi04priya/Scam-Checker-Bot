"""Changes an operator can make from the admin pages: limits, blocked users, fact-check sources."""

import re
from dataclasses import dataclass
from typing import Final

from scambot.api.admin.deps import AdminDeps
from scambot.core.config import FactcheckSource, SourcesConfig
from scambot.core.logging import get_logger
from scambot.core.runtime import (
    CACHE_WINDOW_KEY,
    MAX_SOURCES,
    RATE_LIMIT_KEY,
    SOURCES_KEY,
    effective_sources,
    merged_sources,
)

_PREFIX_RE: Final = re.compile(r"^[0-9a-f]{8,64}$")
_log = get_logger(component="admin")


class ControlError(Exception):
    """A refused change; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class _Limit:
    key: str
    low: float
    high: float
    integer: bool


LIMITS: Final = {
    "rate_limit_per_hour": _Limit(RATE_LIMIT_KEY, 1, 1000, True),
    "cache_window_days": _Limit(CACHE_WINDOW_KEY, 0.01, 365, False),
}


def _log_action(action: str, key: str, old: object, new: object) -> None:
    _log.info("admin_action", action=action, key=key, old=old, new=new)


def check_limit(name: str, raw: str) -> int | float | None:
    """The validated value of a limit field; None for blank (clear the override)."""
    limit = LIMITS[name]
    text = raw.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ControlError(f"{name} must be a number") from None
    if not (limit.low <= number <= limit.high) or (limit.integer and number != int(number)):
        kind = "a whole number" if limit.integer else "a number"
        raise ControlError(f"{name} must be {kind} from {limit.low:g} to {limit.high:g}")
    return int(number) if limit.integer else number


async def set_limit(deps: AdminDeps, name: str, raw: str) -> None:
    """Set a limit, or clear the override when ``raw`` is blank."""
    limit = LIMITS[name]
    value = check_limit(name, raw)
    old = (await deps.runtime.describe())[limit.key]["value"]
    if value is None:
        await deps.settings_store.delete(limit.key)
        await deps.runtime.invalidate(limit.key)
        _log_action("limit_cleared", limit.key, old, None)
        return
    await deps.settings_store.set(limit.key, value)
    await deps.runtime.invalidate(limit.key)
    _log_action("limit_set", limit.key, old, value)


async def set_blocked(deps: AdminDeps, prefix: str, blocked: bool) -> None:
    prefix = prefix.strip().lower()
    if not _PREFIX_RE.match(prefix):
        raise ControlError("the user ID needs at least 8 hex characters")
    found = await deps.stats.find_users(prefix)
    if not found:
        raise ControlError("no such user", 404)
    if len(found) > 1:
        raise ControlError("several users match; type more characters", 409)
    user_hash = found[0]
    await deps.settings_store.set_blocked(user_hash, blocked)
    await deps.runtime.invalidate(f"blocked:{user_hash}")
    _log_action("block" if blocked else "unblock", "user", prefix[:12], blocked)


# --- sources -------------------------------------------------------------------------------


def _as_json(sources: list[FactcheckSource]) -> list[dict[str, object]]:
    return [source.model_dump(mode="json") for source in sources]


async def current_sources(deps: AdminDeps) -> tuple[SourcesConfig, bool]:
    """The publishers in effect and whether they come from a saved override."""
    baseline = deps.config.sources
    effective = await effective_sources(deps.settings_store, baseline)
    return effective, effective is not baseline


async def _save_sources(deps: AdminDeps, sources: list[FactcheckSource], action: str) -> None:
    try:
        merged = merged_sources(deps.config.sources, sources)
    except ValueError as exc:
        raise ControlError(_first_error(exc)) from None
    await deps.settings_store.set(SOURCES_KEY, _as_json(list(merged.sources)))
    _log_action(action, SOURCES_KEY, None, len(merged.sources))


def _first_error(exc: ValueError) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        first = errors()[0]
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "sources")
        return f"{where}: {first['msg']}" if where else str(first["msg"])
    return str(exc)


async def add_source(  # noqa: PLR0913
    deps: AdminDeps, *, source_id: str, name: str, site: str, language: str, api: bool, feeds: str
) -> None:
    sources, _ = await current_sources(deps)
    existing = list(sources.sources)
    if len(existing) >= MAX_SOURCES:
        raise ControlError(f"at most {MAX_SOURCES} publishers are allowed")
    try:
        new = FactcheckSource.model_validate(
            {
                "id": source_id.strip(),
                "name": name.strip(),
                "site": site.strip().lower(),
                "language": language.strip().lower(),
                "api": api,
                "feeds": [line.strip() for line in feeds.splitlines() if line.strip()],
            }
        )
    except ValueError as exc:
        raise ControlError(_first_error(exc)) from None
    await _save_sources(deps, [*existing, new], "source_add")


async def remove_source(deps: AdminDeps, source_id: str) -> None:
    sources, _ = await current_sources(deps)
    kept = [source for source in sources.sources if source.id != source_id]
    if len(kept) == len(sources.sources):
        raise ControlError("no such publisher", 404)
    await _save_sources(deps, kept, "source_remove")


async def reset_sources(deps: AdminDeps) -> None:
    await deps.settings_store.delete(SOURCES_KEY)
    _log_action("source_reset", SOURCES_KEY, None, None)
