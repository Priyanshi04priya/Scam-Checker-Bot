"""Async SQLAlchemy engine and session factory."""

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine as _create_async_engine


def create_engine(url: SecretStr) -> AsyncEngine:
    return _create_async_engine(url.get_secret_value(), pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
