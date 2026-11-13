"""Finding an earlier verdict that can be shown again, on SQLite and in memory."""

import datetime as dt
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.contracts import CheckerResult, ExecutedCheck, Verdict
from scambot.core.db import create_engine, create_session_factory
from scambot.core.models import Base, Request, VerdictRow
from scambot.core.store import (
    Fingerprint,
    NewRequest,
    RequestStatus,
    RequestStore,
    SqlRequestStore,
    StoredResult,
    hamming,
)
from tests.fakes import InMemoryStore

NOW = dt.datetime(2026, 11, 13, 12, 0, tzinfo=dt.UTC)
WINDOW = dt.timedelta(days=7)
SHA = "a" * 64
CAPTION = "c" * 64
PHASH = 0x0F0F_0F0F_0F0F_0F0F
LOOKUP = {"window": WINDOW, "max_distance": 8, "candidates": 500, "now": NOW}


def fp(**changes: object) -> Fingerprint:
    base = {"kind": "image", "sha256": SHA, "phash": PHASH, "caption_hash": CAPTION}
    return Fingerprint(**{**base, **changes})  # type: ignore[arg-type]


def checker(status: str = "ok") -> ExecutedCheck:
    result = CheckerResult(
        checker="check_provenance",
        status=status,  # type: ignore[arg-type]
        signal="none",
        confidence=0.5,
        duration_ms=1,
    )
    return ExecutedCheck(result=result, input_hash="h" * 64)


def sessions_of(store: RequestStore) -> async_sessionmaker[AsyncSession]:
    assert isinstance(store, SqlRequestStore)
    return store._sessions


@pytest.fixture(params=["sqlite", "memory"])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[RequestStore]:
    if request.param == "memory":
        yield InMemoryStore()
        return
    engine = create_engine(SecretStr(f"sqlite+aiosqlite:///{tmp_path}/t.db"))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield SqlRequestStore(create_session_factory(engine))
    await engine.dispose()


async def add(
    store: RequestStore,
    job: str,
    *,
    print_: Fingerprint | None = None,
    status: RequestStatus = "replied",
    age: dt.timedelta = dt.timedelta(hours=1),
    checks: tuple[ExecutedCheck, ...] = (),
    verdict: bool = True,
    user: str = "u1",
) -> uuid.UUID:
    print_ = print_ or fp()
    started = await store.start(
        NewRequest(job, user, "telegram", print_.kind, print_.sha256, "en", print_.caption_hash)
    )
    await store.save(
        started.id,
        StoredResult(
            checks=list(checks),
            verdict=Verdict(label="likely_scam", band="high", score=90, deciding_checker="x")
            if verdict
            else None,
            reply_text="reply text",
            reply_language="en",
            claims=[],
            language="en",
            cost_usd=Decimal("0.5"),
            sha256=print_.sha256,
            phash=print_.phash,
        ),
    )
    await store.set_status(started.id, status, at=NOW - age)
    return started.id


async def find(store: RequestStore, print_: Fingerprint | None = None, **extra: object) -> object:
    args = {**LOOKUP, "exclude": uuid.uuid4(), **extra}
    return await store.find_cached(print_ or fp(), **args)  # type: ignore[arg-type]


def test_hamming_works_for_signed_values() -> None:
    assert hamming(0, 0) == 0
    assert hamming(-1, 0) == 64
    assert hamming(-(2**63), 2**63 - 1) == 64
    assert hamming(0b1011, 0b0001) == 2


async def test_exact_match_is_found(store: RequestStore) -> None:
    first = await add(store, "j1")

    hit = await find(store)

    assert hit is not None
    assert (hit.request_id, hit.match, hit.distance) == (first, "exact", 0)  # type: ignore[attr-defined]
    assert hit.reply_text == "reply text"  # type: ignore[attr-defined]
    assert hit.label == "likely_scam"  # type: ignore[attr-defined]


async def test_the_current_request_is_excluded(store: RequestStore) -> None:
    first = await add(store, "j1")

    assert await find(store, exclude=first) is None


async def test_a_different_caption_is_a_miss(store: RequestStore) -> None:
    await add(store, "j1")

    assert await find(store, fp(caption_hash="d" * 64, phash=None)) is None


async def test_a_missing_caption_matches_a_missing_caption(store: RequestStore) -> None:
    await add(store, "j1", print_=fp(caption_hash=None))

    assert await find(store, fp(caption_hash=None)) is not None
    assert await find(store, fp()) is None


async def test_a_different_kind_is_a_miss(store: RequestStore) -> None:
    await add(store, "j1", print_=fp(kind="video", phash=None))

    assert await find(store, fp(phash=None)) is None


async def test_text_matches_by_hash(store: RequestStore) -> None:
    text = Fingerprint("text", SHA, None, None)
    await add(store, "j1", print_=text)

    assert await find(store, text) is not None


async def test_the_newest_match_wins(store: RequestStore) -> None:
    await add(store, "old", age=dt.timedelta(days=3))
    newer = await add(store, "new", age=dt.timedelta(hours=2))

    hit = await find(store)

    assert hit is not None
    assert hit.request_id == newer  # type: ignore[attr-defined]


async def test_rows_outside_the_window_are_not_served(store: RequestStore) -> None:
    await add(store, "j1", age=dt.timedelta(days=8))

    assert await find(store) is None


@pytest.mark.parametrize(
    ("distance", "found"), [(0, True), (1, True), (8, True), (9, False), (20, False)]
)
async def test_phash_distance_limit(store: RequestStore, distance: int, found: bool) -> None:
    await add(store, "j1")
    flipped = PHASH ^ ((1 << distance) - 1)

    hit = await find(store, fp(sha256="b" * 64, phash=flipped))

    assert (hit is not None) is found
    if hit is not None:
        assert (hit.match, hit.distance) == ("phash", distance)  # type: ignore[attr-defined]


async def test_phash_works_for_negative_stored_values(store: RequestStore) -> None:
    negative = -(2**63) + 5
    await add(store, "j1", print_=fp(phash=negative))

    hit = await find(store, fp(sha256="b" * 64, phash=negative ^ 0b111))

    assert hit is not None
    assert hit.distance == 3  # type: ignore[attr-defined]


async def test_phash_needs_the_same_caption(store: RequestStore) -> None:
    await add(store, "j1")

    assert await find(store, fp(sha256="b" * 64, caption_hash="d" * 64)) is None


async def test_the_closest_phash_wins(store: RequestStore) -> None:
    await add(store, "far", print_=fp(sha256="1" * 64, phash=PHASH ^ 0b1111))
    close = await add(store, "near", print_=fp(sha256="2" * 64, phash=PHASH ^ 0b1))

    hit = await find(store, fp(sha256="b" * 64))

    assert hit is not None
    assert hit.request_id == close  # type: ignore[attr-defined]


async def test_only_images_match_by_phash(store: RequestStore) -> None:
    await add(store, "j1", print_=fp(kind="video"))

    assert await find(store, fp(kind="video", sha256="b" * 64)) is None


async def test_a_verdict_made_during_an_outage_is_not_served(store: RequestStore) -> None:
    await add(store, "j1", checks=(checker("ok"), checker("error")))

    assert await find(store) is None


async def test_inconclusive_checks_are_still_served(store: RequestStore) -> None:
    await add(store, "j1", checks=(checker("inconclusive"),))

    assert await find(store) is not None


@pytest.mark.parametrize("status", ["timeout", "failed", "received"])
async def test_unfinished_requests_are_not_served(store: RequestStore, status: str) -> None:
    await add(store, "j1", status=status)  # type: ignore[arg-type]

    assert await find(store) is None


async def test_a_request_without_a_verdict_is_not_served(store: RequestStore) -> None:
    await add(store, "j1", verdict=False)

    assert await find(store) is None


async def test_save_cached_copies_the_verdict_and_costs_nothing(store: RequestStore) -> None:
    first = await add(store, "j1")
    hit = await find(store)
    assert hit is not None
    second = await store.start(NewRequest("j2", "u2", "telegram", "image", None, "en", CAPTION))

    await store.save_cached(second.id, hit, fp(sha256="b" * 64))  # type: ignore[arg-type]
    await store.set_status(second.id, "replied", at=NOW)

    if isinstance(store, InMemoryStore):
        row = store.requests["j2"]
        assert (row.cache_of, row.cost_usd, row.sha256) == (first, 0, "b" * 64)
        assert row.verdict is not None
        assert row.reply_text == "reply text"
    else:
        async with sessions_of(store)() as session:
            request = (
                await session.execute(select(Request).where(Request.job_id == "j2"))
            ).scalar_one()
            verdict = await session.get(VerdictRow, request.id)
        assert (request.cache_of, request.cost_usd, request.sha256) == (first, Decimal(0), "b" * 64)
        assert verdict is not None
        assert (verdict.label, verdict.reply_text) == ("likely_scam", "reply text")
    # A reused verdict is never itself reused: the original stays the source.
    again = await find(store, fp(sha256="b" * 64, phash=None))
    assert again is None


async def test_load_results_rebuilds_the_saved_checks(store: RequestStore) -> None:
    first = await add(store, "j1", checks=(checker("ok"),))

    (result,) = await store.load_results(first)

    assert (result.checker, result.status) == ("check_provenance", "ok")


def test_the_postgres_distance_expression_binds_a_64_bit_integer() -> None:
    from sqlalchemy import BigInteger, bindparam, cast, func  # noqa: PLC0415
    from sqlalchemy.dialects import postgresql  # noqa: PLC0415
    from sqlalchemy.dialects.postgresql import BIT  # noqa: PLC0415

    negative = -(2**63) + 5
    expression = func.bit_count(
        cast(Request.phash.op("#")(bindparam("probe", negative, type_=BigInteger)), BIT(64))
    )
    compiled = expression.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
    assert "bit_count(CAST(requests.phash # %(probe)s::BIGINT AS BIT(64)))" in str(compiled)
    assert compiled.params["probe"] == negative
