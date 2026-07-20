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
    # How long the track was actually listened to, reported by the bot once it
    # stops playing. NULL means "we handed this over but never heard back" — it
    # was queued and probably never reached the speakers.
    #
    # This is the line between the two signals: every row counts for "don't offer
    # this again right now", but only rows with played_ms count as taste.
    played_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Full length of the track, so the share actually listened to can be worked
    # out later. Not used yet.
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # finished / skipped / stopped / disconnected — stored as-is. Deliberately not
    # interpreted: a skip does not mean dislike, people skip songs they love.
    played_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # "when did this track recently play here" — the freshness lookup.
        Index("ix_play_events_guild_track_time", "guild_id", "track_id", "played_at"),
        # "what has this server been listening to lately" — taste profiles later.
        Index("ix_play_events_guild_time", "guild_id", "played_at"),
    )


class ArtistTags(Base):
    """Genre tags for an artist, as told to us by the search service.

    This is a cache, and it is also how listening history gets its genres: a play
    event stores the artist, and the genres are looked up through here. Storing
    tags on the play row instead would mean fetching them while the user waits
    (~0.8s per artist) and repeating the same tags on every play of the track.

    Tags barely ever change, so rows are refreshed only when they get old.
    """

    __tablename__ = "artist_tags"

    # The artist name exactly as we asked for it — usually the messy YouTube
    # uploader ("Death From Above 1979 - Topic"). The search service cleans it up
    # on its side; we cache under what we sent so lookups always hit.
    artist: Mapped[str] = mapped_column(Text, primary_key=True)
    # [{"name": "nu metal", "weight": 100}, ...]. Empty list = we asked and the tag
    # source genuinely doesn't know this artist; don't keep asking.
    tags: Mapped[Any] = mapped_column(JSONB, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
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
