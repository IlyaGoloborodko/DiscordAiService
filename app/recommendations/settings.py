"""All the knobs for the recommendation system, in one place.

Everything is read from the environment when it is used, not at import time,
because `.env` is loaded after this module gets imported. Change a value in
`.env`, restart the service, done — no code changes needed.
"""

import os


def _number(name: str, default: float) -> float:
    """Read a number from the environment, falling back to `default` if it is
    missing or someone typed something that isn't a number."""
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def freshness_halflife_hours() -> float:
    """After this long, a HEARD play counts for half as much toward holding a
    track back (then a quarter, an eighth...). Sets how long a track you actually
    listened to stays quieter before drifting back to normal."""
    return _number("FRESHNESS_HALFLIFE_HOURS", 24)


def freshness_halflife_unheard_hours() -> float:
    """Same, for a track that was only queued and never actually heard. Short on
    purpose: it just needs to not be offered twice while it's still in the queue."""
    return _number("FRESHNESS_HALFLIFE_UNHEARD_HOURS", 2)


def freshness_weight() -> float:
    """How hard staleness pushes a track down. 0 = ignore play history entirely;
    higher = a recently or often played track is held back harder."""
    return max(0.0, _number("FRESHNESS_WEIGHT", 1.0))


def freshness_floor_minutes() -> float:
    """A track that played this recently is held out completely (weight 0) —
    unless every candidate is, in which case the rule is relaxed."""
    return _number("FRESHNESS_ABSOLUTE_MIN_MINUTES", 45)


def artist_separation_weight() -> float:
    """How hard to avoid several tracks by the same artist in one selection.
    0 = don't care; higher = spread artists out more."""
    return max(0.0, _number("ARTIST_SEPARATION_WEIGHT", 1.0))


def pool_factor() -> float:
    """We ask the search service for this many times more tracks than we need,
    then choose from the bigger pile. A bigger pile means more variety."""
    return max(1.0, _number("CANDIDATE_POOL_FACTOR", 4))


def rank_bias() -> float:
    """How strongly we still prefer the search service's top results.

    High (3+): almost always the top hits — predictable, repetitive.
    Around 1.5: top hits are likelier, but the rest gets a real chance.
    0: complete lottery, ignores how relevant the result was.
    """
    return max(0.0, _number("CANDIDATE_RANK_BIAS", 1.5))
