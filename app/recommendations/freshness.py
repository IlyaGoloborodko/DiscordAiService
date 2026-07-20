"""Deciding how much to hold a track back because it played recently.

People re-listen to music they like — repeats are fine, hearing the SAME thing
too soon is not. So we don't ban a track; we make it less likely to be picked the
more (and more recently) it has played, and the effect fades on its own: a track
left alone drifts back to full chances.

Two signals turn out to be one sum. Every play gets a score that fades with age —
0.5 after one half-life, 0.25 after two, and so on. Add them up:

  * the most recent play alone answers "did it just play?" (its score is ~1);
  * several recent plays pile up and answer "has it played a lot lately?".

That single number is `staleness`. High staleness → low pick weight. A play from
last week weighs almost nothing, so a track that behaved returns to normal by
itself — no counters to reset.

Heard and merely-queued plays fade at different speeds. A track you actually
listened to stays quieter for about a day; one that only sat in a queue unheard
just needs to not be offered twice while it's still there, so it fades in hours.

On top of the soft weight there is one hard rule (`within_floor`): a track that
played in the last few minutes is held out completely. The caller drops that rule
only when EVERY candidate is inside it — better a repeat than nothing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.recommendations import settings


@dataclass
class Play:
    """One time a track was handed to the player. `heard` is True only if we
    later got a confirmation that it actually reached the speakers."""

    played_at: datetime
    heard: bool


def _as_utc(when: datetime) -> datetime:
    # Postgres can hand back a naive timestamp; treat it as UTC rather than crash.
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def staleness(plays: list[Play], now: datetime | None = None) -> float:
    """How "worn out" a track is right now: the sum of every play's fading score.

    0 means no recent record of it. It grows with how often and how recently the
    track played, and decays back to 0 on its own as those plays age.
    """
    now = now or datetime.now(timezone.utc)
    total = 0.0
    for play in plays:
        halflife = (
            settings.freshness_halflife_hours()
            if play.heard
            else settings.freshness_halflife_unheard_hours()
        )
        age_hours = max(0.0, (now - _as_utc(play.played_at)).total_seconds() / 3600.0)
        total += 0.5 ** (age_hours / halflife)
    return total


def soft_multiplier(plays: list[Play], now: datetime | None = None) -> float:
    """A factor in (0, 1] to multiply a track's pick weight by. 1 = untouched,
    smaller = held back. Never quite 0 — a track is only ever made less likely,
    not impossible. Making it impossible is the hard floor's job."""
    return 1.0 / (1.0 + settings.freshness_weight() * staleness(plays, now))


def within_floor(plays: list[Play], now: datetime | None = None) -> bool:
    """True if the track played so recently it must not come back yet at all."""
    now = now or datetime.now(timezone.utc)
    floor = timedelta(minutes=settings.freshness_floor_minutes())
    return any(now - _as_utc(play.played_at) < floor for play in plays)
