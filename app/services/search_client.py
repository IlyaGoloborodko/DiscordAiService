import logging
import os

import httpx

from app.data.models import Track

logger = logging.getLogger(__name__)


class SearchClient:
    """Thin async client over the search-service (yt-dlp lives there, not here).

    Only search is needed in this service: the agent picks tracks by id, and the
    Discord bot resolves stream URLs just-in-time before playback.

    Every call is BEST-EFFORT: an unreachable or failing search-service yields an
    empty list, never an exception. These run as agent tools, so a raised error
    would propagate out of `/agent` as a 500 — the bot would get nothing and the
    user would hear silence. Empty results instead let the agent fall back to
    another tool, and the response guard turns a delivered-nothing turn into an
    honest reply."""

    def __init__(self) -> None:
        self.base_url = os.getenv("SEARCH_SERVICE_URL", "http://127.0.0.1:9000").rstrip("/")

    async def _get(self, path: str, params: dict[str, object], timeout: int) -> list[Track]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self.base_url}{path}", params=params)
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError):
            logger.warning("search-service %s failed (params=%s)", path, params, exc_info=True)
            return []

        tracks = []
        for entry in data.get("results", []):
            try:
                tracks.append(Track(**entry))
            except (TypeError, ValueError):
                # One malformed entry must not lose the whole result set.
                logger.warning("skipping malformed track from %s: %r", path, entry)
        return tracks

    async def search(self, query: str, limit: int = 10, provider: str = "youtube") -> list[Track]:
        return await self._get(
            "/search", {"q": query, "provider": provider, "limit": limit}, timeout=30
        )

    async def playlist(self, url: str, limit: int = 50, provider: str = "youtube") -> list[Track]:
        return await self._get(
            "/playlist", {"url": url, "provider": provider, "limit": limit}, timeout=60
        )

    async def similar(
        self, artist: str, track: str | None = None, limit: int = 10
    ) -> list[Track]:
        """Last.fm-backed recommendations: tracks similar to an artist (+ track)."""
        return await self._get(
            "/similar", self._params(artist=artist, track=track, limit=limit), timeout=60
        )

    async def charts(
        self, tag: str | None = None, country: str | None = None, limit: int = 10
    ) -> list[Track]:
        """Last.fm-backed charts: top tracks for a tag (genre/mood) and/or country;
        omitting both yields the global top."""
        return await self._get(
            "/charts", self._params(tag=tag, country=country, limit=limit), timeout=60
        )

    async def tags(self, artist: str, track: str | None = None, limit: int = 10) -> list[dict]:
        """Genre/style tags for an artist (or a specific track), strongest first.

        Returns entries like {"name": "nu metal", "weight": 100}. An artist the tag
        source doesn't know gives an empty list, not an error — the search service
        also cleans up messy YouTube names ("Death From Above 1979 - Topic") before
        looking them up.
        """
        params = self._params(artist=artist, track=track, limit=limit)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{self.base_url}/tags", params=params)
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError):
            logger.warning("search-service /tags failed (params=%s)", params, exc_info=True)
            return []

        return [tag for tag in data.get("tags", []) if isinstance(tag, dict) and tag.get("name")]

    @staticmethod
    def _params(**kwargs: object) -> dict[str, object]:
        """Drop None values so optional query params are omitted rather than sent
        as empty strings (httpx keeps None as ``key=``, which is not omission)."""
        return {key: value for key, value in kwargs.items() if value is not None}
