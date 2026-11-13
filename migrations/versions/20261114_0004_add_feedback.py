"""Add the feedback table and the request duration.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-14 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("user_hash", sa.String(length=64), nullable=False),
        sa.Column("helpful", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["requests.id"],
            name=op.f("fk_feedback_request_id_requests"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("request_id", "user_hash", name=op.f("pk_feedback")),
    )
    op.create_index("ix_feedback_created_at", "feedback", ["created_at"])
    op.add_column("requests", sa.Column("duration_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.drop_column("duration_ms")
    op.drop_index("ix_feedback_created_at", table_name="feedback")
    op.drop_table("feedback")
