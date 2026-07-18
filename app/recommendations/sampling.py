"""Choosing which of the found tracks to actually offer.

The search service is deterministic: ask it for "metal" today and tomorrow and
you get the same list in the same order. If we always hand over the first few
results, the bot plays the same songs forever — which is exactly the complaint
this module exists to fix.

So we ask for a bigger pile than we need and pick from it at random. Not a flat
lottery though: results near the top are more relevant, so they get a better
chance. `rank_bias` controls how much better.
"""

import random

from app.data.models import Track
from app.recommendations import settings


def pool_size(wanted: int) -> int:
    """How many tracks to request so there is something to choose between."""
    return max(wanted, int(wanted * settings.pool_factor()))


def _weights(count: int, bias: float) -> list[float]:
    """A chance for each position in the list: first place gets the most.

    The weight for position i is 1 / (i + 1) ** bias, so with bias 1.5 the first
    result is about 2.8x likelier than the third. With bias 0 every position is
    equally likely.
    """
    return [1.0 / (index + 1) ** bias for index in range(count)]


def pick_varied(tracks: list[Track], wanted: int, rng: random.Random | None = None) -> list[Track]:
    """Pick `wanted` tracks out of `tracks`, favouring the ones ranked higher.

    Keeps the original order of whatever it picked, so the most relevant track
    still tends to come first in what the bot plays.
    """
    if wanted <= 0 or not tracks:
        return []
    if len(tracks) <= wanted:
        return tracks

    rng = rng or random
    weights = _weights(len(tracks), settings.rank_bias())

    remaining = list(range(len(tracks)))
    chosen: list[int] = []
    for _ in range(wanted):
        weight_of_remaining = [weights[i] for i in remaining]
        position = rng.choices(range(len(remaining)), weights=weight_of_remaining, k=1)[0]
        chosen.append(remaining.pop(position))

    return [tracks[index] for index in sorted(chosen)]
