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


def cooldown_base_hours() -> float:
    """How long a track rests after being played once."""
    return _number("PLAY_COOLDOWN_BASE_HOURS", 6)


def cooldown_growth() -> float:
    """How much longer the rest gets each time the track is played again.
    2.0 means it doubles: 6h, then 12h, then 24h..."""
    return _number("PLAY_COOLDOWN_GROWTH", 2.0)


def cooldown_max_hours() -> float:
    """The rest never grows past this, so a favourite is never banned forever."""
    return _number("PLAY_COOLDOWN_MAX_HOURS", 336)  # two weeks


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
