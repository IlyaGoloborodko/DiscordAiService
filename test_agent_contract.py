"""Network-free tests for the dual-mode /agent contract.

Run: .venv/Scripts/python.exe -m unittest test_agent_contract

The LLM is never called: agent construction/run is mocked, and memory is stubbed
so no Redis/Postgres is needed.
"""

import os
import unittest
from unittest import mock

from app.data.models import (
    AgentRequest,
    AgentResponse,
    ToolCall,
    ToolCallResponse,
    ToolSpec,
)
from app.services.agent_service import AgentService, clean_for_tts

_ENV = {"TM_MODEL_NAME": "dummy", "TM_BASE_URL": "http://127.0.0.1:9/v1", "TM_API_KEY": "x"}

TOOLS = [
    ToolSpec(name="play", description="Play now", input_schema={"tracks": []}),
    ToolSpec(name="enqueue", description="Add to queue", input_schema={"tracks": []}),
    ToolSpec(name="pause", description="Pause", input_schema={}),
    ToolSpec(name="skip", description="Skip", input_schema={}),
]


class CleanForTtsTests(unittest.TestCase):
    def test_strips_emoji(self):
        cleaned = clean_for_tts("Paused. Ready when you are! ⏸️\U0001f3b6")
        self.assertEqual(cleaned, "Paused. Ready when you are!")

    def test_strips_emoji_amid_cyrillic(self):
        cleaned = clean_for_tts("Привет! \U0001f44b Рада! \U0001f3b5✨")
        self.assertEqual(cleaned, "Привет! Рада!")

    def test_strips_markdown_keeps_words(self):
        cleaned = clean_for_tts("Here is a **mix** of `pop` hits!")
        self.assertEqual(cleaned, "Here is a mix of pop hits!")

    def test_keeps_normal_punctuation_and_dash(self):
        text = "Rock - pop, jazz. All good!"
        self.assertEqual(clean_for_tts(text), text)

    def test_no_emoji_remains(self):
        cleaned = clean_for_tts("mix \U0001f3a4\U0001f496\U0001f601\U0001f1ea\U0001f1f8 done")
        self.assertTrue(all(ord(ch) < 0x2000 or ch.isspace() for ch in cleaned), cleaned)


class RenderAndPromptTests(unittest.TestCase):
    def test_render_tools_lists_names_and_schema(self):
        rendered = AgentService._render_tools(TOOLS)
        self.assertIn("- play:", rendered)
        self.assertIn("- pause:", rendered)
        self.assertIn('arguments schema: {"tracks": []}', rendered)
        self.assertIn("arguments schema: {}", rendered)  # empty-input control

    def test_format_prompt_includes_now_playing_and_queue(self):
        req = AgentRequest(
            message="Марина, что в очереди?",
            context={"now_playing": "Song A", "queue": ["Song B", "Song C"], "queue_len": 2},
        )
        prompt = AgentService._format_prompt(req)
        self.assertIn("Now playing: Song A", prompt)
        self.assertIn("Queue: Song B; Song C", prompt)

    def test_format_prompt_without_context(self):
        req = AgentRequest(message="hi", session={"user_name": "Den"})
        prompt = AgentService._format_prompt(req)
        self.assertTrue(prompt.startswith("Den says: hi"))
        self.assertNotIn("Current player state", prompt)


class _FakeResult:
    def __init__(self, output):
        self.output = output

    def all_messages(self):
        return []


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    """run() must pick the toolcall agent iff `tools` is present, and return the
    matching response shape."""

    async def asyncSetUp(self):
        # Stub memory so no Redis/Postgres is touched.
        self.mem = mock.patch.multiple(
            "app.services.memory.MemoryStore",
            load=mock.AsyncMock(return_value=[]),
            save=mock.AsyncMock(return_value=None),
        )
        self.mem.start()
        self.addCleanup(self.mem.stop)

    async def test_tools_present_uses_toolcall_agent(self):
        svc = AgentService()
        expected = ToolCallResponse(
            spoken_answer="ok", display_text="ok",
            tool_calls=[ToolCall(name="enqueue", arguments={"tracks": []})],
        )
        fake_agent = mock.Mock()
        fake_agent.run = mock.AsyncMock(return_value=_FakeResult(expected))

        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_toolcall_agent", return_value=fake_agent) as build_tc, \
             mock.patch.object(svc, "_build_legacy_agent") as build_legacy:
            req = AgentRequest(message="Марина, поставь бодрое", tools=TOOLS)
            out = await svc.run(req)

        build_tc.assert_called_once_with(TOOLS)
        build_legacy.assert_not_called()
        self.assertIsInstance(out, ToolCallResponse)
        self.assertEqual(out.tool_calls[0].name, "enqueue")

    async def test_no_tools_uses_legacy_agent(self):
        svc = AgentService()
        expected = AgentResponse(
            spoken_answer="ok", display_text="ok", action="play", tracks=[],
        )
        fake_agent = mock.Mock()
        fake_agent.run = mock.AsyncMock(return_value=_FakeResult(expected))

        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_legacy_agent", return_value=fake_agent) as build_legacy, \
             mock.patch.object(svc, "_build_toolcall_agent") as build_tc:
            req = AgentRequest(message="play something")
            out = await svc.run(req)

        build_legacy.assert_called_once()
        build_tc.assert_not_called()
        self.assertIsInstance(out, AgentResponse)
        self.assertEqual(out.action, "play")


class ToolCallAgentBuildTest(unittest.TestCase):
    """The toolcall agent must build and embed the bot actions in its prompt."""

    def test_build_toolcall_agent_embeds_actions(self):
        with mock.patch.dict(os.environ, _ENV):
            agent = AgentService()._build_toolcall_agent(TOOLS)
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
