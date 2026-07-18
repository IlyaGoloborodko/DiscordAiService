import os
from datetime import datetime
from typing import Any

from sqlalchemy import TIMESTAMP, BigInteger, Index, Text, func
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


class PlayEvent(Base):
    """One track that was actually sent to the player, with who asked for it.

    This is the durable listening history. It answers "how often has this played
    and when was the last time" (used to rest a track for a while before it can
    play again), and later "what does this server like" for recommendations.

    Written when a track is DELIVERED to the bot, not when search finds it — a
    track nobody heard should not count as taste.
    """

    __tablename__ = "play_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    guild_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    track_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    uploader: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The search query this track came from ("metal", "phonk"). A rough stand-in
    # for genre until the search service can give us real Last.fm tags.
    source_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    # play / enqueue / replace_queue — how it reached the player.
    action: Mapped[str | None] = mapped_column(Text, nullable=True)
    played_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    # How long it was actually listened to. Nothing fills this in yet; it exists
    # so that switching from "times played" to "minutes listened" later needs no
    # second migration.
    played_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    __table_args__ = (
        # "has this track played recently, and how often" — the cooldown lookup.
        Index("ix_play_events_guild_track_time", "guild_id", "track_id", "played_at"),
        # "what has this server been listening to lately" — taste profiles later.
        Index("ix_play_events_guild_time", "guild_id", "played_at"),
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
