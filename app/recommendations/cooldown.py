"""Deciding when a track is allowed to play again.

People re-listen to music they like — that is normal and good. What annoys them
is hearing the same track *too soon*. So we don't ban repeats; we make a track
rest for a while after it plays, and the rest gets longer every time it comes
back. A song you've heard once returns in a few hours; one you've heard ten
times stays away for days.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.recommendations import settings


@dataclass
class PlayStat:
    """How well we already know one track: how often it played and when last."""

    play_count: int
    last_played_at: datetime


def rest_hours(play_count: int) -> float:
    """How many hours a track should rest, given how often it has played.

    With the defaults (base 6h, doubling): 1 play -> 6h, 2 -> 12h, 3 -> 24h,
    and so on, until it stops growing at the maximum.
    """
    if play_count <= 0:
        return 0.0
    hours = settings.cooldown_base_hours() * settings.cooldown_growth() ** (play_count - 1)
    return min(hours, settings.cooldown_max_hours())


def is_resting(stat: PlayStat, now: datetime | None = None) -> bool:
    """True if this track played too recently and should sit this one out."""
    now = now or datetime.now(timezone.utc)
    last = stat.last_played_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return now - last < timedelta(hours=rest_hours(stat.play_count))


def resting_track_ids(stats: dict[str, PlayStat], now: datetime | None = None) -> set[str]:
    """Of the tracks we know about, which ones are still resting right now."""
    now = now or datetime.now(timezone.utc)
    return {track_id for track_id, stat in stats.items() if is_resting(stat, now)}
