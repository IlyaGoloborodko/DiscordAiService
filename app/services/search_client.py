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
