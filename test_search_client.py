"""Network-free tests for the search-service discovery client + tool wiring.

Run: .venv/Scripts/python.exe -m unittest test_search_client

Uses httpx.MockTransport so nothing leaves the process; the search-service and
the LLM are never contacted.
"""

import os
import unittest
from unittest import mock

import httpx

from app.services.search_client import SearchClient

# Two tracks in the exact shape the search-service returns from every endpoint.
SAMPLE_RESULTS = [
    {
        "provider": "youtube",
        "id": "abc123",
        "title": "Song A",
        "uploader": "Artist A",
        "url": "https://www.youtube.com/watch?v=abc123",
        "duration": 200.0,
        "thumbnail": None,
    },
    {
        "provider": "youtube",
        "id": "def456",
        "title": "Song B",
        "uploader": "Artist B",
        "url": "https://www.youtube.com/watch?v=def456",
        "duration": 180.0,
        "thumbnail": None,
    },
]

_real_async_client = httpx.AsyncClient


def _mock_client(handler):
    """Patch target: build a real AsyncClient but route requests to `handler`,
    capturing the request so the test can assert the outgoing query params."""

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _real_async_client(*args, **kwargs)

    return mock.patch("app.services.search_client.httpx.AsyncClient", side_effect=factory)


class DiscoveryClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests: list[httpx.Request] = []

    def _handler(self, payload, status=200):
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(status, json=payload)

        return handler

    # --- similar() -----------------------------------------------------------

    async def test_similar_happy_path(self):
        handler = self._handler({"results": SAMPLE_RESULTS})
        with _mock_client(handler):
            tracks = await SearchClient().similar("Daft Punk", "Get Lucky", limit=5)

        self.assertEqual([t.id for t in tracks], ["abc123", "def456"])
        self.assertTrue(self.requests[0].url.path.endswith("/similar"))
        params = self.requests[0].url.params
        self.assertEqual(params["artist"], "Daft Punk")
        self.assertEqual(params["track"], "Get Lucky")
        self.assertEqual(params["limit"], "5")

    async def test_similar_omits_absent_track(self):
        handler = self._handler({"results": []})
        with _mock_client(handler):
            tracks = await SearchClient().similar("Daft Punk")

        self.assertEqual(tracks, [])
        self.assertNotIn("track", self.requests[0].url.params)

    async def test_similar_error_status_raises(self):
        handler = self._handler({"detail": "Last.fm key not configured"}, status=503)
        with _mock_client(handler):
            with self.assertRaises(httpx.HTTPStatusError):
                await SearchClient().similar("Daft Punk")

    # --- charts() ------------------------------------------------------------

    async def test_charts_happy_path_with_tag(self):
        handler = self._handler({"results": SAMPLE_RESULTS})
        with _mock_client(handler):
            tracks = await SearchClient().charts(tag="phonk", limit=2)

        self.assertEqual(len(tracks), 2)
        self.assertTrue(self.requests[0].url.path.endswith("/charts"))
        params = self.requests[0].url.params
        self.assertEqual(params["tag"], "phonk")
        self.assertNotIn("country", params)

    async def test_charts_global_omits_both(self):
        handler = self._handler({"results": SAMPLE_RESULTS})
        with _mock_client(handler):
            await SearchClient().charts()

        params = self.requests[0].url.params
        self.assertNotIn("tag", params)
        self.assertNotIn("country", params)

    async def test_charts_empty_results(self):
        handler = self._handler({"results": []})
        with _mock_client(handler):
            tracks = await SearchClient().charts(country="Japan")

        self.assertEqual(tracks, [])
        self.assertEqual(self.requests[0].url.params["country"], "Japan")

    async def test_charts_upstream_error_raises(self):
        handler = self._handler({"detail": "provider failure"}, status=502)
        with _mock_client(handler):
            with self.assertRaises(httpx.HTTPStatusError):
                await SearchClient().charts(tag="rock")


class ToolWiringSmokeTest(unittest.TestCase):
    """The new tools must register on the agent without error."""

    def test_agent_registers_discovery_tools(self):
        # Dummy env so the model/provider construct without touching the network.
        env = {
            "TM_MODEL_NAME": "dummy",
            "TM_BASE_URL": "http://127.0.0.1:9/v1",
            "TM_API_KEY": "x",
        }
        from app.services.agent_service import AgentService

        with mock.patch.dict(os.environ, env):
            agent = AgentService()._build_agent()

        names = self._tool_names(agent)
        if names is None:
            self.skipTest("agent tool introspection not available in this pydantic-ai version")
        for expected in (
            "search_tracks",
            "get_playlist_tracks",
            "get_similar_tracks",
            "get_top_charts",
        ):
            self.assertIn(expected, names)

    @staticmethod
    def _tool_names(agent) -> set[str] | None:
        toolset = getattr(agent, "_function_toolset", None)
        tools = getattr(toolset, "tools", None)
        if isinstance(tools, dict):
            return set(tools)
        return None


if __name__ == "__main__":
    unittest.main()
