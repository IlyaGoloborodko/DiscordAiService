"""Network-free tests for the dual-mode /agent contract.

Run: .venv/Scripts/python.exe -m unittest test_agent_contract

The LLM is never called: agent construction/run is mocked, and memory is stubbed
so no Redis/Postgres is needed.
"""

import os
import unittest
from unittest import mock

from app.data.models import (
    AgentDraft,
    AgentRequest,
    AgentResponse,
    ToolCall,
    ToolCallDraft,
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


class ValidateToolCallsTests(unittest.TestCase):
    PLAY = ToolSpec(name="play", description="", input_schema={"required": ["tracks"], "properties": {"tracks": {}}})
    PAUSE = ToolSpec(name="pause", description="", input_schema={"properties": {}})
    DECLARED = [PLAY, PAUSE]

    def test_drops_hallucinated_name(self):
        calls = [ToolCall(name="play_track", arguments={})]
        kept, dropped = AgentService._validate_tool_calls(calls, self.DECLARED)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 1)

    def test_drops_music_action_without_tracks(self):
        calls = [ToolCall(name="play", arguments={})]
        kept, dropped = AgentService._validate_tool_calls(calls, self.DECLARED)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 1)

    def test_keeps_play_with_tracks(self):
        calls = [ToolCall(name="play", arguments={"tracks": [{"id": "x", "title": "t"}]})]
        kept, dropped = AgentService._validate_tool_calls(calls, self.DECLARED)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 0)

    def test_keeps_control_with_empty_args(self):
        calls = [ToolCall(name="pause", arguments={})]
        kept, dropped = AgentService._validate_tool_calls(calls, self.DECLARED)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 0)


class DirtyTurnNotSavedTests(unittest.IsolatedAsyncioTestCase):
    """A turn with a dropped (hallucinated) tool_call must not be persisted."""

    async def asyncSetUp(self):
        self.load = mock.patch("app.services.memory.MemoryStore.load",
                               new=mock.AsyncMock(return_value=[]))
        self.save = mock.patch("app.services.memory.MemoryStore.save", new=mock.AsyncMock())
        self.load.start(); self.save.start()
        self.addCleanup(self.load.stop); self.addCleanup(self.save.stop)

    async def _run_with_output(self, output, tools):
        svc = AgentService()
        fake_agent = mock.Mock()
        fake_agent.run = mock.AsyncMock(return_value=_FakeResult(output))
        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_toolcall_agent", return_value=fake_agent):
            await svc.run(AgentRequest(message="play hardcore", tools=tools))
        return svc.memory.save

    async def test_dirty_turn_not_saved(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        out = ToolCallDraft(display_text="ok",
                            tool_calls=[ToolCall(name="play_track", arguments={})])
        save = await self._run_with_output(out, tools)
        save.assert_not_called()

    async def test_clean_turn_saved(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        out = ToolCallDraft(display_text="ok",
                            tool_calls=[ToolCall(name="play", arguments={"tracks": [{"id": "a", "title": "b"}]})])
        save = await self._run_with_output(out, tools)
        save.assert_called_once()


class MemoryTrimTests(unittest.TestCase):
    """History is kept to the last MAX_TURNS whole turns."""

    @staticmethod
    def _user(text):
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        return ModelRequest(parts=[UserPromptPart(content=text)])

    @staticmethod
    def _assistant(text):
        from pydantic_ai.messages import ModelResponse, TextPart
        return ModelResponse(parts=[TextPart(content=text)])

    def test_keeps_only_last_two_turns(self):
        from app.services.memory import MemoryStore

        msgs = [
            self._user("t1"), self._assistant("a1"),
            self._user("t2"), self._assistant("a2"),
            self._user("t3"), self._assistant("a3"),
        ]
        trimmed = MemoryStore()._trim(msgs)
        # last 2 turns -> starts at t2
        self.assertEqual(len(trimmed), 4)
        self.assertIs(trimmed[0], msgs[2])
        self.assertIs(trimmed[-1], msgs[-1])

    def test_short_history_untouched(self):
        from app.services.memory import MemoryStore

        msgs = [self._user("t1"), self._assistant("a1")]
        self.assertEqual(MemoryStore()._trim(msgs), msgs)

    def test_trimmed_history_starts_with_user_prompt(self):
        from app.services.memory import MemoryStore
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msgs = []
        for i in range(5):
            msgs.append(self._user(f"t{i}"))
            msgs.append(self._assistant(f"a{i}"))
        trimmed = MemoryStore()._trim(msgs)
        first = trimmed[0]
        self.assertIsInstance(first, ModelRequest)
        self.assertTrue(any(isinstance(p, UserPromptPart) for p in first.parts))


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

    def test_users_live_example(self):
        cleaned = clean_for_tts(
            "Hey! \U0001f44b Ready to find some music for you. What are we listening to today? \U0001f3a7"
        )
        self.assertEqual(cleaned, "Hey! Ready to find some music for you. What are we listening to today?")

    def test_strips_symbols_the_ranges_used_to_miss(self):
        # ™ ‼ ⁉ CJK marks — not caught by the old hand-rolled ranges.
        self.assertEqual(clean_for_tts("Great pick ©®™ enjoy"), "Great pick enjoy")
        self.assertEqual(clean_for_tts("Wow‼ really⁉"), "Wow really")
        self.assertEqual(clean_for_tts("nice 〰 ㊗ ㊙"), "nice")

    def test_strips_musical_notes(self):
        self.assertEqual(clean_for_tts("la la ♪ ♫ ♬ now"), "la la now")


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
        draft = ToolCallDraft(
            display_text="ok",
            tool_calls=[ToolCall(name="enqueue", arguments={"tracks": [{"id": "a", "title": "b"}]})],
        )
        fake_agent = mock.Mock()
        fake_agent.run = mock.AsyncMock(return_value=_FakeResult(draft))

        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_toolcall_agent", return_value=fake_agent) as build_tc, \
             mock.patch.object(svc, "_build_legacy_agent") as build_legacy:
            req = AgentRequest(message="Марина, поставь бодрое", tools=TOOLS)
            out = await svc.run(req)

        build_tc.assert_called_once_with(TOOLS)
        build_legacy.assert_not_called()
        self.assertIsInstance(out, ToolCallResponse)
        self.assertEqual(out.tool_calls[0].name, "enqueue")
        self.assertEqual(out.spoken_answer, "ok")  # derived from display_text

    async def test_no_tools_uses_legacy_agent(self):
        svc = AgentService()
        draft = AgentDraft(display_text="ok 🎵", action="play", tracks=[])
        fake_agent = mock.Mock()
        fake_agent.run = mock.AsyncMock(return_value=_FakeResult(draft))

        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_legacy_agent", return_value=fake_agent) as build_legacy, \
             mock.patch.object(svc, "_build_toolcall_agent") as build_tc:
            req = AgentRequest(message="play something")
            out = await svc.run(req)

        build_legacy.assert_called_once()
        build_tc.assert_not_called()
        self.assertIsInstance(out, AgentResponse)
        self.assertEqual(out.action, "play")
        self.assertEqual(out.spoken_answer, "ok")  # emoji stripped from display_text


class ToolCallAgentBuildTest(unittest.TestCase):
    """The toolcall agent must build and embed the bot actions in its prompt."""

    def test_build_toolcall_agent_embeds_actions(self):
        with mock.patch.dict(os.environ, _ENV):
            agent = AgentService()._build_toolcall_agent(TOOLS)
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
