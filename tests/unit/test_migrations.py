"""The migrations create exactly the tables the ORM models describe, and can be reverted."""

from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from scambot.core.models import Base
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
            }

        command.downgrade(config, "base")

        with engine.connect() as connection:
            assert set(inspect(connection).get_table_names()) == {"alembic_version"}
    finally:
        engine.dispose()
