"""Create the users, requests, checker_runs and verdicts tables.

Revision ID: 0001
Revises:
Create Date: 2026-11-12 00:00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    """Apply this revision."""
    op.create_table(
        "users",
        sa.Column("user_hash", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("blocked", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("user_hash", name=op.f("pk_users")),
    )
    op.create_table(
        "requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.String(length=128), nullable=False),
        sa.Column("user_hash", sa.String(length=64), nullable=False),
        sa.Column("channel", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("phash", sa.BigInteger(), nullable=True),
        sa.Column("caption_hash", sa.String(length=64), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("claims", JSON_TYPE, nullable=False),
        sa.Column("status", sa.String(length=16), server_default="received", nullable=False),
        sa.Column("cache_of", sa.Uuid(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(10, 6), server_default=sa.text("0"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("replied_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('received', 'replied', 'failed', 'timeout')",
            name=op.f("ck_requests_status"),
        ),
        sa.ForeignKeyConstraint(
            ["cache_of"],
            ["requests.id"],
            name=op.f("fk_requests_cache_of_requests"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_hash"],
            ["users.user_hash"],
            name=op.f("fk_requests_user_hash_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_requests")),
        sa.UniqueConstraint("job_id", name=op.f("uq_requests_job_id")),
    )
    op.create_index(op.f("ix_requests_created_at"), "requests", ["created_at"])
    op.create_index("ix_requests_sha256_caption_hash", "requests", ["sha256", "caption_hash"])
    op.create_index("ix_requests_user_hash_created_at", "requests", ["user_hash", "created_at"])
    op.create_table(
        "checker_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("checker", sa.String(length=64), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("signal", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence", JSON_TYPE, nullable=False),
        sa.Column("details", JSON_TYPE, nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["requests.id"],
            name=op.f("fk_checker_runs_request_id_requests"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_checker_runs")),
    )
    op.create_index(op.f("ix_checker_runs_request_id"), "checker_runs", ["request_id"])
    op.create_table(
        "verdicts",
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("band", sa.String(length=16), nullable=False),
        sa.Column("score", sa.SmallInteger(), nullable=False),
        sa.Column("deciding_checker", sa.String(length=64), nullable=True),
        sa.Column("reply_text", sa.Text(), nullable=False),
        sa.Column("reply_language", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["requests.id"],
            name=op.f("fk_verdicts_request_id_requests"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("request_id", name=op.f("pk_verdicts")),
    )


def downgrade() -> None:
    """Revert this revision."""
    op.drop_table("verdicts")
    op.drop_index(op.f("ix_checker_runs_request_id"), table_name="checker_runs")
    op.drop_table("checker_runs")
    op.drop_index("ix_requests_user_hash_created_at", table_name="requests")
    op.drop_index("ix_requests_sha256_caption_hash", table_name="requests")
    op.drop_index(op.f("ix_requests_created_at"), table_name="requests")
    op.drop_table("requests")
    op.drop_table("users")
