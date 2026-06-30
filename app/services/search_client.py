import os

import httpx

from app.data.models import Track


class SearchClient:
    """Thin async client over the search-service (yt-dlp lives there, not here).

    Only search is needed in this service: the agent picks tracks by id, and the
    Discord bot resolves stream URLs just-in-time before playback."""

    def __init__(self) -> None:
        self.base_url = os.getenv("SEARCH_SERVICE_URL", "http://127.0.0.1:9000").rstrip("/")

    async def search(self, query: str, limit: int = 10, provider: str = "youtube") -> list[Track]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.base_url}/search",
                params={"q": query, "provider": provider, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()

        return [Track(**entry) for entry in data.get("results", [])]

    async def playlist(self, url: str, limit: int = 50, provider: str = "youtube") -> list[Track]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"{self.base_url}/playlist",
                params={"url": url, "provider": provider, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()

        return [Track(**entry) for entry in data.get("results", [])]

    async def similar(
        self, artist: str, track: str | None = None, limit: int = 10
    ) -> list[Track]:
        """Last.fm-backed recommendations: tracks similar to an artist (+ track)."""
        params = self._params(artist=artist, track=track, limit=limit)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(f"{self.base_url}/similar", params=params)
            resp.raise_for_status()
            data = resp.json()

        return [Track(**entry) for entry in data.get("results", [])]

    async def charts(
        self, tag: str | None = None, country: str | None = None, limit: int = 10
    ) -> list[Track]:
        """Last.fm-backed charts: top tracks for a tag (genre/mood) and/or country;
        omitting both yields the global top."""
        params = self._params(tag=tag, country=country, limit=limit)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(f"{self.base_url}/charts", params=params)
            resp.raise_for_status()
            data = resp.json()

        return [Track(**entry) for entry in data.get("results", [])]

    @staticmethod
    def _params(**kwargs: object) -> dict[str, object]:
        """Drop None values so optional query params are omitted rather than sent
        as empty strings (httpx keeps None as ``key=``, which is not omission)."""
        return {key: value for key, value in kwargs.items() if value is not None}
