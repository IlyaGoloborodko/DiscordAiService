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


def pick_varied(
    tracks: list[Track],
    wanted: int,
    rng: random.Random | None = None,
    soft: dict[str, float] | None = None,
    floored: set[str] | None = None,
) -> list[Track]:
    """Pick `wanted` tracks out of `tracks`, favouring the ones ranked higher.

    Three things bend the odds, all multiplied together:
      * rank — a higher search result gets a better chance (`rank_bias`);
      * freshness — `soft[id]` (a factor in (0,1]) holds back tracks that played
        recently, and `floored` ids are held out entirely (see recommendations
        .freshness);
      * artist spread — once a track by some artist is picked, other tracks by
        the same artist get a worse chance, so one artist can't fill the set.

    Keeps the original order of whatever it picked, so the most relevant track
    still tends to come first in what the bot plays.
    """
    if wanted <= 0 or not tracks:
        return []
    if len(tracks) <= wanted:
        return tracks

    rng = rng or random
    soft = soft or {}
    floored = floored or set()
    base = _weights(len(tracks), settings.rank_bias())
    gamma = settings.artist_separation_weight()

    remaining = list(range(len(tracks)))
    chosen: list[int] = []
    picked_by_artist: dict[str, int] = {}

    def artist_factor(i: int) -> float:
        artist = (tracks[i].uploader or "").lower()
        already = picked_by_artist.get(artist, 0) if artist else 0
        return 1.0 / (1.0 + gamma * already)

    for _ in range(wanted):
        # With the hard floor first; if that zeroes every remaining track, relax
        # it (better a repeat than nothing) but keep the soft weighting so the
        # least-recently-played still wins.
        soft_weights = [base[i] * soft.get(tracks[i].id, 1.0) * artist_factor(i) for i in remaining]
        weights = [0.0 if tracks[i].id in floored else w for i, w in zip(remaining, soft_weights)]
        if not any(weights):
            weights = soft_weights

        position = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        index = remaining.pop(position)
        chosen.append(index)
        artist = (tracks[index].uploader or "").lower()
        if artist:
            picked_by_artist[artist] = picked_by_artist.get(artist, 0) + 1

    return [tracks[index] for index in sorted(chosen)]
