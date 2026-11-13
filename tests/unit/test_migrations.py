"""The migrations create exactly the tables the ORM models describe, and can be reverted."""

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from scambot.core.models import Base, IndexBase
from scambot.core.settings import PROJECT_ROOT


def test_upgrade_matches_models_and_downgrade_reverts(tmp_path: Path) -> None:
    database = tmp_path / "migrations.db"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{database}")

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database}")
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_type": True, "compare_server_default": True}
            )
            assert compare_metadata(context, Base.metadata) == []
            assert set(inspect(connection).get_table_names()) == {
                "alembic_version",
                "users",
                "requests",
                "checker_runs",
                "verdicts",
                "settings",
                "feedback",
            }

        command.downgrade(config, "base")

        with engine.connect() as connection:
            assert set(inspect(connection).get_table_names()) == {"alembic_version"}
    finally:
        engine.dispose()


def test_offline_sql_creates_the_fact_check_index(capsys: pytest.CaptureFixture[str]) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", "postgresql+asyncpg://u@h/db")

    command.upgrade(config, "head", sql=True)

    sql = capsys.readouterr().out
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "embedding VECTOR(384) NOT NULL" in sql
    assert "CREATE TABLE feedback" in sql
    assert "ALTER TABLE requests ADD COLUMN duration_ms INTEGER" in sql
    assert "GENERATED ALWAYS AS (to_tsvector('simple'::regconfig" in sql
    assert "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)" in sql
    assert "USING gin (tsv)" in sql
    assert "CREATE TABLE settings" in sql
    assert "CREATE INDEX ix_requests_kind_replied_at" in sql


POSTGRES_URL = os.environ.get("SCAMBOT_TEST_DATABASE_URL", "")


async def _reset_postgres() -> None:
    engine = create_async_engine(POSTGRES_URL)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(IndexBase.metadata.drop_all)
            await connection.run_sync(Base.metadata.drop_all)
            await connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
    finally:
        await engine.dispose()


async def _postgres_differences() -> tuple[list[Any], set[str]]:
    def compare(connection: Connection) -> tuple[list[Any], set[str]]:
        context = MigrationContext.configure(
            connection, opts={"compare_type": True, "compare_server_default": True}
        )
        # compare_metadata accepts a list of MetaData, as env.py's target_metadata does.
        metadata: Any = [Base.metadata, IndexBase.metadata]
        differences = compare_metadata(context, metadata)
        return differences, set(inspect(connection).get_table_names())

    engine = create_async_engine(POSTGRES_URL)
    try:
        async with engine.connect() as connection:
            return await connection.run_sync(compare)
    finally:
        await engine.dispose()


@pytest.mark.skipif(not POSTGRES_URL, reason="SCAMBOT_TEST_DATABASE_URL is not set")
def test_postgres_upgrade_matches_both_model_sets() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", POSTGRES_URL)
    asyncio.run(_reset_postgres())
    command.upgrade(config, "head")

    differences, tables = asyncio.run(_postgres_differences())
    assert differences == []
    assert "factcheck_articles" in tables

    command.downgrade(config, "0001")
    _, tables = asyncio.run(_postgres_differences())
    assert "factcheck_articles" not in tables
    command.upgrade(config, "head")
