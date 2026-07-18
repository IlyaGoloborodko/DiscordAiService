"""Reading and writing the listening history.

Two jobs:
  * remember that a track was played (so we can rest it, and learn taste later);
  * look up how often the tracks we know about have played, and when.

Like the rest of our storage, this is best-effort: if Postgres is down the bot
keeps working, it just forgets. Losing history is annoying; failing a request
because of it would be worse.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.data.models import Track
from app.recommendations import settings
from app.recommendations.cooldown import PlayStat
from app.storage import PlayEvent, get_sessionmaker

logger = logging.getLogger(__name__)


async def record_plays(
    guild_id: str,
    user_id: str | None,
    tracks: list[Track],
    action: str,
    source_queries: dict[str, str] | None = None,
) -> None:
    """Write down the tracks we just handed to the player."""
    if not guild_id or not tracks:
        return

    source_queries = source_queries or {}
    rows = [
        PlayEvent(
            guild_id=guild_id,
            user_id=user_id,
            track_id=track.id,
            title=track.title,
            uploader=track.uploader,
            provider=track.provider,
            source_query=source_queries.get(track.id),
            action=action,
        )
        for track in tracks
    ]

    try:
        async with get_sessionmaker()() as session:
            session.add_all(rows)
            await session.commit()
    except Exception:
        logger.warning("could not record %d plays for %s", len(rows), guild_id, exc_info=True)


async def load_play_stats(guild_id: str) -> dict[str, PlayStat]:
    """How often each track has played on this server, and when it last did.

    Only tracks played inside the longest possible rest are returned — anything
    older cannot be resting any more, so there is no point loading it. That
    keeps this query small no matter how long the history grows.
    """
    if not guild_id:
        return {}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.cooldown_max_hours())
    query = (
        select(
            PlayEvent.track_id,
            func.count().label("play_count"),
            func.max(PlayEvent.played_at).label("last_played_at"),
        )
        .where(PlayEvent.guild_id == guild_id)
        .group_by(PlayEvent.track_id)
        .having(func.max(PlayEvent.played_at) >= cutoff)
    )

    try:
        async with get_sessionmaker()() as session:
            rows = (await session.execute(query)).all()
    except Exception:
        logger.warning("could not load play stats for %s", guild_id, exc_info=True)
        return {}

    return {
        row.track_id: PlayStat(play_count=row.play_count, last_played_at=row.last_played_at)
        for row in rows
    }
