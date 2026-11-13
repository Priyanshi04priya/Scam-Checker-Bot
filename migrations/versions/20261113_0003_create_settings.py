"""Create the settings table and the index used to find a reusable earlier verdict.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-13 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "settings",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column(
            "value", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_settings")),
    )
    op.create_index("ix_requests_kind_replied_at", "requests", ["kind", "replied_at"])


def downgrade() -> None:
    op.drop_index("ix_requests_kind_replied_at", table_name="requests")
    op.drop_table("settings")
