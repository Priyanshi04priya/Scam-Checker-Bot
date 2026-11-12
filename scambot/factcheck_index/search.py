"""Hybrid search over the fact-check index: nearest embeddings plus full-text matches, fused
by reciprocal rank."""

import asyncio
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from sqlalchemy import Text, cast, func, literal, select
from sqlalchemy.dialects.postgresql import REGCONFIG, TSQUERY, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from scambot.core.models import FactcheckArticle
from scambot.factcheck_index.embed import Embedder
from scambot.factcheck_index.records import ArticleRecord, canonical_url

DEFAULT_RRF_K: Final = 60
CHUNK: Final = 500
MAX_QUERY_CHARS: Final = 500

_TOKEN_SPLIT: Final = re.compile(r"[\s.,;:!?\"'()\[\]{}<>/\\|।॥…“”‘’\-–—]+")  # noqa: RUF001


@dataclass(frozen=True)
class EmbeddedArticle:
    record: ArticleRecord
    embedding: list[float] = field(repr=False)


class FactcheckIndex(Protocol):
    async def nearest(self, embedding: Sequence[float], k: int) -> list[ArticleRecord]: ...

    async def full_text(self, query: str, k: int) -> list[ArticleRecord]: ...

    async def upsert(self, items: Sequence[EmbeddedArticle]) -> int: ...

    async def known_urls(self, urls: Sequence[str]) -> set[str]: ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """Keys ordered by ``Σ 1/(k + rank)`` (ranks from 1); ties go to the better best rank,
    then to the smaller key. A key repeated inside one list counts once, at its first rank."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best[key] = min(best.get(key, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best[item[0]], item[0]))


def fuse(rankings: Sequence[Sequence[ArticleRecord]], k: int) -> list[ArticleRecord]:
    """Records from every ranking, deduplicated by canonical URL and ordered by RRF.

    The first record seen for a URL is kept, unless a later one has a rating it lacks.
    """
    chosen: dict[str, ArticleRecord] = {}
    keys: list[list[str]] = []
    for ranking in rankings:
        ranked: list[str] = []
        for record in ranking:
            key = canonical_url(record.url)
            ranked.append(key)
            current = chosen.get(key)
            if current is None or (current.rating is None and record.rating is not None):
                chosen[key] = record
        keys.append(ranked)
    return [chosen[key] for key, _ in reciprocal_rank_fusion(keys, k)]


@dataclass(frozen=True)
class Retrieval:
    vector: list[ArticleRecord]
    text: list[ArticleRecord]


async def retrieve(index: FactcheckIndex, embedder: Embedder, claim: str, top_k: int) -> Retrieval:
    """The claim's nearest articles by embedding and its best full-text matches."""
    [embedding] = await embedder.embed_queries([claim])
    vector, text = await asyncio.gather(
        index.nearest(embedding, top_k), index.full_text(claim, top_k)
    )
    return Retrieval(vector=vector, text=text)


# --- PostgreSQL ----------------------------------------------------------------------------

_COLUMNS: Final = (
    FactcheckArticle.source,
    FactcheckArticle.url,
    FactcheckArticle.title,
    FactcheckArticle.claim,
    FactcheckArticle.rating,
    FactcheckArticle.published_at,
    FactcheckArticle.language,
)


def _record(row: Any) -> ArticleRecord:
    return ArticleRecord(
        source=row.source,
        url=row.url,
        title=row.title,
        claim=row.claim,
        rating=row.rating,
        published_at=row.published_at,
        language=row.language,
    )


def _chunks[T](items: Sequence[T]) -> list[Sequence[T]]:
    return [items[i : i + CHUNK] for i in range(0, len(items), CHUNK)]


class PgFactcheckIndex:
    """The index in PostgreSQL: pgvector HNSW for embeddings, a GIN index for full text."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def nearest(self, embedding: Sequence[float], k: int) -> list[ArticleRecord]:
        statement = (
            select(*_COLUMNS)
            .order_by(FactcheckArticle.embedding.cosine_distance(list(embedding)))
            .limit(k)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return [_record(row) for row in rows]

    async def full_text(self, query: str, k: int) -> list[ArticleRecord]:
        if not query.strip():
            return []
        # Any word may match (OR); plainto_tsquery strips operators from the input first.
        tsquery = select(
            cast(
                func.replace(
                    cast(
                        func.plainto_tsquery(
                            literal("simple", type_=REGCONFIG), query[:MAX_QUERY_CHARS]
                        ),
                        Text,
                    ),
                    " & ",
                    " | ",
                ),
                TSQUERY,
            )
        ).scalar_subquery()
        statement = (
            select(*_COLUMNS)
            .where(FactcheckArticle.tsv.op("@@")(tsquery))
            .order_by(func.ts_rank(FactcheckArticle.tsv, tsquery).desc())
            .limit(k)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return [_record(row) for row in rows]

    async def upsert(self, items: Sequence[EmbeddedArticle]) -> int:
        if not items:
            return 0
        async with self._sessions() as session:
            for chunk in _chunks(items):
                statement = insert(FactcheckArticle).values(
                    [
                        {
                            "id": uuid.uuid4(),
                            **item.record.model_dump(),
                            "url": canonical_url(item.record.url),
                            "embedding": item.embedding,
                        }
                        for item in chunk
                    ]
                )
                excluded = statement.excluded
                table = FactcheckArticle.__table__.c
                statement = statement.on_conflict_do_update(
                    index_elements=[FactcheckArticle.url],
                    set_={
                        "source": excluded.source,
                        "title": excluded.title,
                        "claim": excluded.claim,
                        "embedding": excluded.embedding,
                        "rating": func.coalesce(excluded.rating, table.rating),
                        "published_at": func.coalesce(excluded.published_at, table.published_at),
                        "language": func.coalesce(excluded.language, table.language),
                        "updated_at": func.now(),
                    },
                )
                await session.execute(statement)
            await session.commit()
        return len(items)

    async def known_urls(self, urls: Sequence[str]) -> set[str]:
        wanted = list(dict.fromkeys(canonical_url(url) for url in urls))
        found: set[str] = set()
        async with self._sessions() as session:
            for chunk in _chunks(wanted):
                statement = select(FactcheckArticle.url).where(FactcheckArticle.url.in_(chunk))
                found.update((await session.execute(statement)).scalars())
        return found


# --- in memory -----------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_SPLIT.split(text.casefold()) if len(token) >= 2}


def _merge(old: ArticleRecord, new: ArticleRecord) -> ArticleRecord:
    return new.model_copy(
        update={
            "rating": new.rating if new.rating is not None else old.rating,
            "published_at": new.published_at or old.published_at,
            "language": new.language or old.language,
        }
    )


@dataclass
class InMemoryFactcheckIndex:
    """The index in a dictionary, for tests and local experiments."""

    items: dict[str, EmbeddedArticle] = field(default_factory=dict)

    async def nearest(self, embedding: Sequence[float], k: int) -> list[ArticleRecord]:
        def score(item: EmbeddedArticle) -> float:
            return sum(a * b for a, b in zip(item.embedding, embedding, strict=True))

        ranked = sorted(self.items.values(), key=score, reverse=True)
        return [item.record for item in ranked[:k]]

    async def full_text(self, query: str, k: int) -> list[ArticleRecord]:
        wanted = _tokens(query)
        scored = [
            (len(wanted & _tokens(f"{item.record.title} {item.record.claim}")), item.record)
            for item in self.items.values()
        ]
        # sorted() is stable, so equal scores keep insertion order.
        ranked = sorted((pair for pair in scored if pair[0] >= 1), key=lambda p: -p[0])
        return [record for _, record in ranked[:k]]

    async def upsert(self, items: Sequence[EmbeddedArticle]) -> int:
        for item in items:
            key = canonical_url(item.record.url)
            record = item.record.model_copy(update={"url": key})
            old = self.items.get(key)
            if old is not None:
                record = _merge(old.record, record)
            self.items[key] = EmbeddedArticle(record, list(item.embedding))
        return len(items)

    async def known_urls(self, urls: Sequence[str]) -> set[str]:
        return {key for key in (canonical_url(url) for url in urls) if key in self.items}
