"""Everything that decides WHICH tracks get played, as opposed to finding them.

Right now it does two things:

  * `cooldown` — a track that just played rests for a while before it can come
    back, and the rest gets longer the more often it has played.
  * `sampling` — the search service always returns the same list for the same
    query, so we ask for more results than we need and pick from them with a
    bias towards the top instead of always taking the first few.

`history` is what both of those are built on: a durable log of what actually
reached the player. Later it also becomes the taste profile (which artists and
genres this server likes), which is why it records more than the cooldown needs.

`genres` fills in what kind of music each artist makes, cached in the database
because looking it up takes about a second and the answer never changes.

All the knobs live in `settings`. See tmp/recommendations-plan.md for the plan
and the reasoning behind it.
"""

from app.recommendations import cooldown, genres, history, sampling, settings

__all__ = ["cooldown", "genres", "history", "sampling", "settings"]
