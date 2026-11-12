"""Keyed hashing of platform user IDs and constant-time secret comparison."""

import hashlib
import hmac

from pydantic import SecretStr


def user_hash(user_id: str | int, pepper: SecretStr) -> str:
    """Return HMAC-SHA256 of ``user_id`` keyed with ``pepper``, as 64 lowercase hex characters.

    The same user always gets the same hash, but the platform ID cannot be recovered from it
    without the pepper.
    """
    key = pepper.get_secret_value().encode()
    return hmac.new(key, str(user_id).encode(), hashlib.sha256).hexdigest()


def secrets_match(received: str | None, expected: SecretStr) -> bool:
    """Compare a received secret with the configured one in constant time.

    An unset (empty) expected secret never matches, so a missing configuration rejects every
    request instead of accepting every request.
    """
    expected_value = expected.get_secret_value()
    if not expected_value or received is None:
        return False
    return hmac.compare_digest(received.encode(), expected_value.encode())
