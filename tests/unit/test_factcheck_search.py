"""Tests for the fact-check index: in memory always, PostgreSQL when a test database is set."""

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import text

from scambot.core.db import create_engine, create_session_factory
from scambot.core.settings import PROJECT_ROOT
from scambot.factcheck_index.records import ArticleRecord
from scambot.factcheck_index.search import (
    EmbeddedArticle,
    InMemoryFactcheckIndex,
    PgFactcheckIndex,
    retrieve,
)
from tests.fakes import KeywordEmbedder, seed
from tests.samples import LAPTOP_CLAIM_HI, LAPTOP_FACTCHECK, UNRELATED_FACTCHECKS

POSTGRES_URL = os.environ.get("SCAMBOT_TEST_DATABASE_URL", "")


async def test_nearest_full_text_and_known_urls(
    factcheck_index: InMemoryFactcheckIndex, embedder: KeywordEmbedder
) -> None:
    await seed(factcheck_index, embedder, LAPTOP_FACTCHECK, *UNRELATED_FACTCHECKS)

    nearest = await factcheck_index.nearest(embedder.vector("free laptops for students"), 2)
    assert nearest[0].url == LAPTOP_FACTCHECK.url
    assert len(nearest) == 2

    found = await factcheck_index.full_text("cricket captain retirement", 5)
    assert [r.source for r in found] == ["altnews"]
    assert await factcheck_index.full_text("nothing matches here zz", 5) == []

    known = await factcheck_index.known_urls(
        [f"{LAPTOP_FACTCHECK.url}?utm_source=wa", "https://example.com/new"]
    )
    assert known == {LAPTOP_FACTCHECK.url}


async def test_upsert_keeps_existing_values_when_the_new_record_lacks_them(
    factcheck_index: InMemoryFactcheckIndex, embedder: KeywordEmbedder
) -> None:
    await seed(factcheck_index, embedder, LAPTOP_FACTCHECK)
    poorer = LAPTOP_FACTCHECK.model_copy(
        update={"rating": None, "published_at": None, "language": None, "title": "New title"}
    )
    await seed(factcheck_index, embedder, poorer)

    (stored,) = [item.record for item in factcheck_index.items.values()]
    assert stored.title == "New title"
    assert (stored.rating, stored.published_at, stored.language) == (
        LAPTOP_FACTCHECK.rating,
        LAPTOP_FACTCHECK.published_at,
        "en",
    )


async def test_retrieve_crosses_languages(
    factcheck_index: InMemoryFactcheckIndex, embedder: KeywordEmbedder
) -> None:
    await seed(factcheck_index, embedder, *UNRELATED_FACTCHECKS, LAPTOP_FACTCHECK)

    found = await retrieve(factcheck_index, embedder, LAPTOP_CLAIM_HI, 2)

    assert embedder.calls[-1] == ("query", [LAPTOP_CLAIM_HI])
    assert found.vector[0].url == LAPTOP_FACTCHECK.url
    assert found.text[0].url == LAPTOP_FACTCHECK.url  # "2026" is shared


def _unit(index: int) -> list[float]:
    vector = [0.0] * 384
    vector[index] = 1.0
    return vector


@pytest.mark.skipif(not POSTGRES_URL, reason="SCAMBOT_TEST_DATABASE_URL is not set")
class TestPostgres:
    @pytest.fixture
    async def pg_index(self) -> AsyncIterator[PgFactcheckIndex]:
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", POSTGRES_URL)
        await asyncio.to_thread(command.upgrade, config, "head")  # env.py runs its own loop
        engine = create_engine(SecretStr(POSTGRES_URL))
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM factcheck_articles"))
        try:
            yield PgFactcheckIndex(create_session_factory(engine))
        finally:
            await engine.dispose()

    async def test_round_trip(self, pg_index: PgFactcheckIndex) -> None:
        hindi = ArticleRecord(
            source="vishvas",
            url="https://www.vishvasnews.com/hindi/laptop/",
            title="फ्री लैपटॉप योजना का दावा फर्जी",
            claim="सरकार सभी छात्रों को मुफ्त लैपटॉप दे रही है",
            language="hi",
        )
        items = [
            EmbeddedArticle(LAPTOP_FACTCHECK, _unit(0)),
            EmbeddedArticle(UNRELATED_FACTCHECKS[0], _unit(1)),
            EmbeddedArticle(hindi, _unit(2)),
        ]
        assert await pg_index.upsert(items) == 3
        poorer = LAPTOP_FACTCHECK.model_copy(update={"rating": None})
        assert await pg_index.upsert([EmbeddedArticle(poorer, _unit(0))]) == 1

        nearest = await pg_index.nearest(_unit(0), 2)
        assert nearest[0].url == LAPTOP_FACTCHECK.url
        assert nearest[0].rating == "False"  # kept by the upsert

        english = await pg_index.full_text("free laptops students", 5)
        assert english[0].url == LAPTOP_FACTCHECK.url
        devanagari = await pg_index.full_text("मुफ्त लैपटॉप", 5)
        assert [r.url for r in devanagari] == [hindi.url]
        assert await pg_index.full_text("   ", 5) == []

        known = await pg_index.known_urls([LAPTOP_FACTCHECK.url, "https://example.com/x"])
        assert known == {LAPTOP_FACTCHECK.url}
