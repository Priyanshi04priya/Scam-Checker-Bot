"""ORM models. Chat identifiers and message text are never stored."""

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    Uuid,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class User(Base):
    __tablename__ = "users"

    user_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    language: Mapped[str | None] = mapped_column(String(16))
    blocked: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Request(Base):
    __tablename__ = "requests"
    __table_args__ = (
        CheckConstraint("status IN ('received', 'replied', 'failed', 'timeout')", name="status"),
        Index("ix_requests_sha256_caption_hash", "sha256", "caption_hash"),
        Index("ix_requests_user_hash_created_at", "user_hash", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[str] = mapped_column(String(128), unique=True)
    user_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_hash", ondelete="CASCADE")
    )
    channel: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16))
    sha256: Mapped[str | None] = mapped_column(String(64))
    phash: Mapped[int | None] = mapped_column(BigInteger)
    caption_hash: Mapped[str | None] = mapped_column(String(64))
    language: Mapped[str | None] = mapped_column(String(16))
    claims: Mapped[list[str]] = mapped_column(JSONType, default=list)
    status: Mapped[str] = mapped_column(String(16), server_default="received")
    cache_of: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("requests.id", ondelete="SET NULL")
    )
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6), server_default=text("0"))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    replied_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class CheckerRun(Base):
    __tablename__ = "checker_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("requests.id", ondelete="CASCADE"), index=True
    )
    checker: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    signal: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONType)
    details: Mapped[dict[str, Any]] = mapped_column(JSONType)
    duration_ms: Mapped[int] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)


class VerdictRow(Base):
    __tablename__ = "verdicts"

    request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("requests.id", ondelete="CASCADE"), primary_key=True
    )
    label: Mapped[str] = mapped_column(String(32))
    band: Mapped[str] = mapped_column(String(16))
    score: Mapped[int] = mapped_column(SmallInteger)
    deciding_checker: Mapped[str | None] = mapped_column(String(64))
    reply_text: Mapped[str] = mapped_column(Text)
    reply_language: Mapped[str] = mapped_column(String(16))
