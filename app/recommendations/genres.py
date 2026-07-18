"""Finding out what genre a track is.

The search service can tell us an artist's tags ("nu metal", "metal", "rock"),
but each lookup costs the better part of a second, so we never do it while the
user is waiting. Instead:

  * tags are cached per artist in the database, effectively forever — an artist's
    genre does not change;
  * the cache is warmed in the background as music plays, so by the time the
    taste profile needs genres they are usually already there;
  * anything still missing is fetched on demand, in parallel.

Genres are attached to the ARTIST, not to each play. That means history recorded
before we knew the genre still gets one later, for free.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.services.search_client import SearchClient
from app.storage import ArtistTags, get_sessionmaker

logger = logging.getLogger(__name__)

# How many tags to keep per artist. The first few carry the genre; the tail is
# noise like "seen live" and the artist's own name.
TAGS_PER_ARTIST = 10


def _refresh_after_days() -> float:
    """How stale a cached entry may get before we ask again."""
    try:
        return float(os.getenv("ARTIST_TAGS_REFRESH_DAYS") or 90)
    except ValueError:
        return 90


async def cached_tags(artists: list[str]) -> dict[str, list[dict]]:
    """Whatever we already know about these artists. Missing ones are simply absent."""
    wanted = [artist for artist in artists if artist]
    if not wanted:
        return {}

    stale_before = datetime.now(timezone.utc) - timedelta(days=_refresh_after_days())
    try:
        async with get_sessionmaker()() as session:
            rows = (
                await session.execute(
                    select(ArtistTags).where(ArtistTags.artist.in_(wanted))
                )
            ).scalars().all()
    except Exception:
        logger.warning("could not read artist tags", exc_info=True)
        return {}

    known = {}
    for row in rows:
        fetched = row.fetched_at
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        if fetched >= stale_before:
            known[row.artist] = row.tags or []
    return known


async def _store(artist: str, tags: list[dict]) -> None:
    statement = pg_insert(ArtistTags).values(artist=artist, tags=tags)
    statement = statement.on_conflict_do_update(
        index_elements=[ArtistTags.artist],
        set_={"tags": tags, "fetched_at": datetime.now(timezone.utc)},
    )
    try:
        async with get_sessionmaker()() as session:
            await session.execute(statement)
            await session.commit()
    except Exception:
        logger.warning("could not cache tags for %r", artist, exc_info=True)


async def fetch_missing(artists: list[str], search: SearchClient | None = None) -> None:
    """Look up the artists we have no tags for and remember what we find.

    Artists the tag source doesn't know are cached as an empty list on purpose,
    so we don't ask about them again on every single play.
    """
    unknown = sorted({artist for artist in artists if artist} - set(await cached_tags(artists)))
    if not unknown:
        return

    search = search or SearchClient()
    results = await asyncio.gather(
        *(search.tags(artist, limit=TAGS_PER_ARTIST) for artist in unknown),
        return_exceptions=True,
    )
    for artist, tags in zip(unknown, results):
        if isinstance(tags, BaseException):
            logger.warning("tag lookup failed for %r: %s", artist, tags)
            continue
        await _store(artist, tags)


def warm_cache_in_background(artists: list[str]) -> None:
    """Start filling in tags for these artists without making anyone wait.

    Called right after tracks are handed to the player. If it fails or never
    finishes, nothing breaks — whatever is missing gets fetched later, when the
    taste profile actually needs it.
    """
    if not artists:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop (scripts, tests) — nothing to schedule onto
    task = loop.create_task(fetch_missing(list(artists)))
    # Without a reference the task can be garbage-collected mid-flight, and
    # without a callback a failure would be swallowed silently.
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


_background_tasks: set[asyncio.Task] = set()


async def genres_for_artists(artists: list[str]) -> dict[str, list[str]]:
    """Genre names per artist, strongest first. Fetches anything not cached yet.

    This is what the taste profile will use, so it is allowed to be slow — it
    runs when we are choosing music, not while the user waits for a reply.
    """
    await fetch_missing(artists)
    known = await cached_tags(artists)
    return {artist: [tag["name"] for tag in tags] for artist, tags in known.items()}
