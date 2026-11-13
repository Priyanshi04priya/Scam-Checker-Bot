"""HTTP Basic login for the admin pages and a request-forgery token for their forms."""

import asyncio
import base64
import binascii
import hashlib
import hmac
import time
from collections.abc import Callable
from functools import cache
from typing import Final
from urllib.parse import urlsplit

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import HTTPException, Request, status

from scambot.core.logging import get_logger

MAX_REMEMBERED: Final = 16
MAX_FAILURES: Final = 10
FAILURE_WINDOW_S: Final = 300
_log = get_logger(component="admin")


@cache
def _dummy_hash() -> str:
    return PasswordHasher().hash("not-the-password")


class AdminAuth:
    """Checks credentials and issues stateless CSRF tokens."""

    def __init__(  # noqa: PLR0913
        self,
        user: str,
        password_hash: str,
        pepper: str,
        *,
        cache_s: int,
        csrf_ttl_s: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._user = user
        self._hash = password_hash
        self._hasher = PasswordHasher()
        self._csrf_key = hashlib.sha256(
            b"admin-csrf" + password_hash.encode() + pepper.encode()
        ).digest()
        self._cache_s = cache_s
        self.csrf_ttl_s = csrf_ttl_s
        self._clock = clock
        self._ok: dict[str, float] = {}

    @property
    def user(self) -> str:
        return self._user

    def _verify(self, user: str, password: str) -> bool:
        user_ok = hmac.compare_digest(user.encode(), self._user.encode())
        # The slow hash always runs so the answer time does not reveal whether the user exists.
        try:
            self._hasher.verify(self._hash if user_ok else _dummy_hash(), password)
        except (VerificationError, InvalidHashError):
            return False
        return user_ok

    async def check(self, header: str | None) -> bool:
        if not header or not header.lower().startswith("basic "):
            return False
        key = hashlib.sha256(header.encode()).hexdigest()
        now = self._clock()
        if self._ok.get(key, 0.0) > now:
            return True
        try:
            decoded = base64.b64decode(header[6:].strip(), validate=True).decode()
        except (binascii.Error, UnicodeDecodeError):
            return False
        user, _, password = decoded.partition(":")
        if not await asyncio.to_thread(self._verify, user, password):
            return False
        if len(self._ok) >= MAX_REMEMBERED:
            self._ok = {k: v for k, v in self._ok.items() if v > now}
            while len(self._ok) >= MAX_REMEMBERED:
                self._ok.pop(next(iter(self._ok)))
        self._ok[key] = now + self._cache_s
        return True

    # --- CSRF ------------------------------------------------------------------------------

    def _sign(self, ts: int) -> str:
        return hmac.new(self._csrf_key, f"{ts}|{self._user}".encode(), hashlib.sha256).hexdigest()

    def make_token(self, now_ts: int) -> str:
        return f"{now_ts}.{self._sign(now_ts)}"

    def token_valid(self, token: str | None, now_ts: int) -> bool:
        if not token:
            return False
        stamp, _, signature = token.partition(".")
        if not (stamp.isascii() and stamp.isdigit() and signature.isascii()):
            return False
        if not hmac.compare_digest(signature.encode(), self._sign(int(stamp)).encode()):
            return False
        return 0 <= now_ts - int(stamp) <= self.csrf_ttl_s


async def _failures(request: Request, *, add: bool) -> int:
    """Failed logins from this connection's address in the window; Redis trouble counts as none.

    The address is the socket peer (the server resolves a proxy's forwarding header only for
    proxies it was told to trust), never a header the caller can set freely.
    """
    key = f"admin:authfail:{request.client.host if request.client else '-'}"
    redis = request.app.state.admin.redis
    try:
        if not add:
            return int(await redis.get(key) or 0)
        count = int(await redis.incr(key))
        if count == 1:
            await redis.expire(key, FAILURE_WINDOW_S)
        return count
    except Exception as exc:  # the login still works, only the throttle is off
        _log.warning("admin_throttle_unavailable", error=type(exc).__name__)
        return 0


async def require_admin(request: Request) -> None:
    auth: AdminAuth = request.app.state.admin_auth
    if await _failures(request, add=False) >= MAX_FAILURES:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, headers={"Retry-After": "300"})
    header = request.headers.get("authorization")
    if not await auth.check(header):
        if header:  # the first request of a browser carries no credentials and is not a guess
            await _failures(request, add=True)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": 'Basic realm="admin"'}
        )


async def require_csrf(request: Request) -> None:
    """Reject a POST without a valid token, or one sent from another site."""
    auth: AdminAuth = request.app.state.admin_auth
    origin = request.headers.get("origin")
    if origin and urlsplit(origin).netloc != request.headers.get("host", ""):
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    now = int(request.app.state.admin.now().timestamp())
    # The page-wide header token goes stale on a long-open page while swapped forms carry a
    # fresh one, so either valid token is accepted.
    if auth.token_valid(request.headers.get("x-csrf-token"), now):
        return
    form = await request.form()
    field_value = form.get("csrf_token")
    if auth.token_valid(field_value if isinstance(field_value, str) else None, now):
        return
    _log.info("csrf_rejected")
    raise HTTPException(status.HTTP_403_FORBIDDEN)
