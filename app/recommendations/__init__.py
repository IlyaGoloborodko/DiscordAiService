"""Everything that decides WHICH tracks get played, as opposed to finding them.

Right now it does two things:

  * `freshness` — a track that played recently is made less likely to be picked,
    more so the more (and more recently) it played, and the effect fades on its
    own. A very recent one is held out entirely for a few minutes.
  * `sampling` — the search service always returns the same list for the same
    query, so we ask for more results than we need and pick from them, biased
    towards the top, held back by `freshness`, and spread across artists.

`history` is what both of those are built on: a durable log of what actually
reached the player. Later it also becomes the taste profile (which artists and
genres this server likes), which is why it records more than freshness needs.

`genres` fills in what kind of music each artist makes, cached in the database
because looking it up takes about a second and the answer never changes.

All the knobs live in `settings`. See tmp/recommendations-plan.md for the plan
and the reasoning behind it.
"""

from app.recommendations import freshness, genres, history, sampling, settings

__all__ = ["freshness", "genres", "history", "sampling", "settings"]
