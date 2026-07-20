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

from sqlalchemy import select

from app.data.models import Track
from app.recommendations import freshness, settings
from app.storage import PlayEvent, get_sessionmaker

logger = logging.getLogger(__name__)

# How many half-lives back to bother loading. Past this a play's fading score is
# under 0.5^8 ≈ 0.4% — it changes nothing, so leaving it out just keeps the query small.
_HORIZON_HALFLIVES = 8


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


async def confirm_play(
    guild_id: str,
    track_id: str,
    played_ms: int,
    duration_ms: int | None = None,
    reason: str | None = None,
    provider: str | None = None,
) -> None:
    """Mark that a track we handed over was actually heard, and for how long.

    Fills in the newest row for this track that is still waiting for confirmation.
    If there isn't one, the track was played without us — someone used the bot's
    own commands instead of asking the bot — and we write a fresh row. That is
    real listening either way, and arguably the most honest kind.
    """
    if not guild_id or not track_id or played_ms <= 0:
        return

    waiting = (
        select(PlayEvent)
        .where(
            PlayEvent.guild_id == guild_id,
            PlayEvent.track_id == track_id,
            PlayEvent.played_ms.is_(None),
        )
        .order_by(PlayEvent.played_at.desc())
        .limit(1)
    )

    try:
        async with get_sessionmaker()() as session:
            row = (await session.execute(waiting)).scalar_one_or_none()
            if row is None:
                # Played outside of the bot, or this track has already been
                # confirmed once and is now playing again — either way, a new
                # listen deserves its own row.
                row = PlayEvent(
                    guild_id=guild_id,
                    track_id=track_id,
                    title=track_id,
                    provider=provider,
                    action="external",
                )
                session.add(row)
            row.played_ms = played_ms
            row.duration_ms = duration_ms
            row.played_reason = reason
            await session.commit()
    except Exception:
        logger.warning("could not confirm play of %s in %s", track_id, guild_id, exc_info=True)


async def load_plays(guild_id: str) -> dict[str, list[freshness.Play]]:
    """Every recent play of every track on this server, grouped by track.

    We return the individual plays, not a count — freshness sums each play's
    fading score on the fly (no stored counters). Only plays inside the window
    where that score still matters are loaded (a few half-lives back); older ones
    weigh ~0 and would just bloat the query. `heard` is True when the bot later
    confirmed the track actually reached the speakers (`played_ms` is set).
    """
    if not guild_id:
        return {}

    horizon = max(settings.freshness_halflife_hours(), settings.freshness_halflife_unheard_hours())
    cutoff = datetime.now(timezone.utc) - timedelta(hours=horizon * _HORIZON_HALFLIVES)
    query = (
        select(PlayEvent.track_id, PlayEvent.played_at, PlayEvent.played_ms)
        .where(PlayEvent.guild_id == guild_id, PlayEvent.played_at >= cutoff)
    )

    try:
        async with get_sessionmaker()() as session:
            rows = (await session.execute(query)).all()
    except Exception:
        logger.warning("could not load plays for %s", guild_id, exc_info=True)
        return {}

    plays: dict[str, list[freshness.Play]] = {}
    for row in rows:
        plays.setdefault(row.track_id, []).append(
            freshness.Play(played_at=row.played_at, heard=row.played_ms is not None)
        )
    return plays
