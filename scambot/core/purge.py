"""Deleting requests with everything that hangs off them, without relying on foreign keys."""

import datetime as dt
import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scambot.core.models import CheckerRun, Feedback, Request, User, VerdictRow

CHUNK = 500
PURGE_BATCH = 1000


async def delete_requests(session: AsyncSession, ids: Sequence[uuid.UUID]) -> None:
    """Remove requests with their runs and verdicts; rows that reused one forget it."""
    for start in range(0, len(ids), CHUNK):
        chunk = list(ids[start : start + CHUNK])
        await session.execute(delete(Feedback).where(Feedback.request_id.in_(chunk)))
        await session.execute(delete(VerdictRow).where(VerdictRow.request_id.in_(chunk)))
        await session.execute(delete(CheckerRun).where(CheckerRun.request_id.in_(chunk)))
        await session.execute(
            update(Request).where(Request.cache_of.in_(chunk)).values(cache_of=None)
        )
        await session.execute(delete(Request).where(Request.id.in_(chunk)))


async def delete_user_rows(session: AsyncSession, user_hash: str) -> int:
    """Remove everything stored for one user; a blocked user keeps the row that blocks them."""
    ids = list(await session.scalars(select(Request.id).where(Request.user_hash == user_hash)))
    await delete_requests(session, ids)
    user = await session.get(User, user_hash)
    if user is not None:
        if user.blocked:
            user.language = None
        else:
            await session.delete(user)
    return len(ids)


async def purge_rows(session: AsyncSession, cutoff: dt.datetime) -> int:
    """Remove requests created before ``cutoff`` in batches, then users left with nothing."""
    removed = 0
    while True:
        ids = list(
            await session.scalars(
                select(Request.id).where(Request.created_at < cutoff).limit(PURGE_BATCH)
            )
        )
        if not ids:
            break
        await delete_requests(session, ids)
        await session.flush()
        removed += len(ids)
    has_requests = select(Request.user_hash).where(Request.user_hash == User.user_hash).exists()
    await session.execute(
        delete(User).where(User.created_at < cutoff, User.blocked.is_(False), ~has_requests)
    )
    return removed
