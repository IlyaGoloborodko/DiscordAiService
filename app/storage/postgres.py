import os
from datetime import datetime
from typing import Any

from sqlalchemy import TIMESTAMP, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base; Alembic reads Base.metadata for migrations."""


class SessionMemory(Base):
    """Serialized pydantic-ai conversation history, one row per session key."""

    __tablename__ = "agent_sessions"

    session_key: Mapped[str] = mapped_column(Text, primary_key=True)
    messages: Mapped[Any] = mapped_column(JSONB, nullable=False)
    # Recently played tracks, newest first, capped. Kept in its own column rather
    # than inside `messages`: the conversation history holds only clean text turns
    # (see AgentService.run), and the Discord bot loses its queue on restart, so the
    # service is the only place "what was playing" can survive.
    recent_tracks: Mapped[Any] = mapped_column(JSONB, nullable=False, server_default="[]")
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def async_dsn() -> str:
    """Normalise POSTGRES_DSN to the asyncpg driver SQLAlchemy expects."""
    dsn = os.getenv("POSTGRES_DSN", "")
    for prefix in ("postgresql+asyncpg://", ):
        if dsn.startswith(prefix):
            return dsn
    if dsn.startswith("postgresql://"):
        return "postgresql+asyncpg://" + dsn[len("postgresql://"):]
    if dsn.startswith("postgres://"):
        return "postgresql+asyncpg://" + dsn[len("postgres://"):]
    return dsn


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Lazily create a process-wide async engine (connection pool inside)."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(async_dsn(), pool_size=5, max_overflow=5, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker
