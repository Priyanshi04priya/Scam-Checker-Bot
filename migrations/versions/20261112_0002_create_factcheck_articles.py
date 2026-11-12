"""Create the factcheck_articles table (PostgreSQL only: needs pgvector).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12 00:00:01

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TSV = "to_tsvector('simple'::regconfig, coalesce(title, '') || ' ' || coalesce(claim, ''))"


def _postgresql() -> bool:
    return bool(op.get_context().dialect.name == "postgresql")


def upgrade() -> None:
    if not _postgresql():
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "factcheck_articles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("rating", sa.String(length=200), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("tsv", postgresql.TSVECTOR(), sa.Computed(TSV, persisted=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_factcheck_articles")),
        sa.UniqueConstraint("url", name=op.f("uq_factcheck_articles_url")),
    )
    op.create_index(
        op.f("ix_factcheck_articles_published_at"),
        "factcheck_articles",
        ["published_at"],
        unique=False,
    )
    op.create_index(
        "ix_factcheck_articles_embedding",
        "factcheck_articles",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_factcheck_articles_tsv",
        "factcheck_articles",
        ["tsv"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    if not _postgresql():
        return
    op.drop_index("ix_factcheck_articles_tsv", table_name="factcheck_articles")
    op.drop_index("ix_factcheck_articles_embedding", table_name="factcheck_articles")
    op.drop_index(op.f("ix_factcheck_articles_published_at"), table_name="factcheck_articles")
    op.drop_table("factcheck_articles")
