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
    ToolCallDraft,
    ToolCallResponse,
    ToolSpec,
    Track,
)
from app.services.agent_service import AgentService, clean_for_tts

_ENV = {"TM_MODEL_NAME": "dummy", "TM_BASE_URL": "http://127.0.0.1:9/v1", "TM_API_KEY": "x"}

TOOLS = [
    ToolSpec(name="play", description="Play now", input_schema={"tracks": []}),
    ToolSpec(name="enqueue", description="Add to queue", input_schema={"tracks": []}),
    ToolSpec(name="pause", description="Pause", input_schema={}),
    ToolSpec(name="skip", description="Skip", input_schema={}),
]


class BuildToolCallsTests(unittest.TestCase):
    PLAY = ToolSpec(name="play", input_schema={"required": ["tracks"]})
    PAUSE = ToolSpec(name="pause", input_schema={})
    TOOLS = [PLAY, PAUSE]
    TRACK = Track(id="x", title="t")

    def test_no_action(self):
        calls, clean = AgentService._build_tool_calls("", [], self.TOOLS)
        self.assertEqual(calls, [])
        self.assertTrue(clean)

    def test_hallucinated_action_is_dirty(self):
        calls, clean = AgentService._build_tool_calls("play_track", [self.TRACK], self.TOOLS)
        self.assertEqual(calls, [])
        self.assertFalse(clean)

    def test_music_action_without_tracks_is_dirty(self):
        calls, clean = AgentService._build_tool_calls("play", [], self.TOOLS)
        self.assertEqual(calls, [])
        self.assertFalse(clean)

    def test_play_with_tracks(self):
        calls, clean = AgentService._build_tool_calls("play", [self.TRACK], self.TOOLS)
        self.assertTrue(clean)
        self.assertEqual(calls[0].name, "play")
        self.assertEqual(calls[0].arguments.tracks[0].id, "x")

    def test_control_needs_no_tracks(self):
        calls, clean = AgentService._build_tool_calls("pause", [], self.TOOLS)
        self.assertTrue(clean)
        self.assertEqual(calls[0].name, "pause")


class DirtyTurnNotSavedTests(unittest.IsolatedAsyncioTestCase):
    """A turn that produced no valid action (hallucinated / no tracks) is not saved."""

    async def asyncSetUp(self):
        self.load = mock.patch("app.services.memory.MemoryStore.load",
                               new=mock.AsyncMock(return_value=[]))
        self.save = mock.patch("app.services.memory.MemoryStore.save", new=mock.AsyncMock())
        self.load.start(); self.save.start()
        self.addCleanup(self.load.stop); self.addCleanup(self.save.stop)

    async def _run(self, draft, tools, found=None):
        svc = AgentService()

        async def fake_run(prompt, deps=None, message_history=None):
            if found:
                deps.found.update(found)
            return _FakeResult(draft)

        fake_agent = mock.Mock()
        fake_agent.run = fake_run
        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_toolcall_agent", return_value=fake_agent):
            resp = await svc.run(AgentRequest(message="play hardcore", tools=tools))
        return svc.memory.save, resp

    async def test_dirty_turn_not_saved(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="ok", action="play_track")
        save, _ = await self._run(draft, tools)
        save.assert_not_called()

    async def test_clean_turn_saved_and_resolves_ids(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="ok", action="play", track_ids=["a"])
        save, resp = await self._run(draft, tools, found={"a": Track(id="a", title="Song")})
        save.assert_called_once()
        self.assertEqual(resp.tool_calls[0].name, "play")
        self.assertEqual(resp.tool_calls[0].arguments.tracks[0].title, "Song")

    async def test_invented_id_dropped(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="ok", action="play", track_ids=["ghost"])
        save, resp = await self._run(draft, tools)  # nothing found -> no real tracks
        self.assertEqual(resp.tool_calls, [])
        save.assert_not_called()


class MemoryTrimTests(unittest.TestCase):
    """Intermediate history is bounded by a token budget; system prompt and the
    latest message are handled elsewhere and never part of this."""

    @staticmethod
    def _user(text):
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        return ModelRequest(parts=[UserPromptPart(content=text)])

    @staticmethod
    def _assistant(text):
        from pydantic_ai.messages import ModelResponse, TextPart
        return ModelResponse(parts=[TextPart(content=text)])

    def _msgs(self, n):
        out = []
        for i in range(n):
            out.append(self._user(f"user message number {i}"))
            out.append(self._assistant(f"assistant reply number {i}"))
        return out

    def test_short_history_untouched(self):
        from app.services.memory import MemoryStore

        msgs = self._msgs(2)
        self.assertEqual(MemoryStore()._trim(msgs), msgs)  # well under 20k tokens

    def test_drops_oldest_when_over_budget(self):
        from app.services.memory import MemoryStore

        store = MemoryStore()
        msgs = self._msgs(10)  # 20 messages
        # Budget for roughly the last 6 messages.
        store.token_limit = store._count_tokens(msgs[-1]) * 6
        trimmed = store._trim(msgs)
        self.assertLess(len(trimmed), len(msgs))
        self.assertGreater(len(trimmed), 0)
        self.assertIs(trimmed[-1], msgs[-1])  # newest always kept

    def test_trimmed_history_starts_with_user_prompt(self):
        from app.services.memory import MemoryStore
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        store = MemoryStore()
        msgs = self._msgs(10)
        store.token_limit = store._count_tokens(msgs[-1]) * 6
        trimmed = store._trim(msgs)
        first = trimmed[0]
        self.assertIsInstance(first, ModelRequest)
        self.assertTrue(any(isinstance(p, UserPromptPart) for p in first.parts))

    def test_strip_system_removes_system_parts(self):
        from app.services.memory import MemoryStore
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        msgs = [
            ModelRequest(parts=[SystemPromptPart(content="SYS")]),
            ModelRequest(parts=[SystemPromptPart(content="SYS"), UserPromptPart(content="hi")]),
            self._assistant("a"),
        ]
        stripped = MemoryStore._strip_system(msgs)
        # the system-only request is dropped; the mixed one keeps just the user part
        self.assertEqual(len(stripped), 2)
        self.assertFalse(
            any(isinstance(p, SystemPromptPart) for m in stripped
                if isinstance(m, ModelRequest) for p in m.parts)
        )


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
    def test_render_tools_lists_names_as_strings(self):
        rendered = AgentService._render_tools(TOOLS)
        # Names are quoted string values, not function-signature-looking entries.
        self.assertIn('"play" (needs tracks)', rendered)
        self.assertIn('"pause"', rendered)
        self.assertNotIn("arguments schema", rendered)

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
        draft = ToolCallDraft(display_text="ok", action="enqueue", track_ids=["a"])

        async def fake_run(prompt, deps=None, message_history=None):
            deps.found["a"] = Track(id="a", title="b")
            return _FakeResult(draft)

        fake_agent = mock.Mock()
        fake_agent.run = fake_run

        with mock.patch.dict(os.environ, _ENV), \
             mock.patch.object(svc, "_build_toolcall_agent", return_value=fake_agent) as build_tc, \
             mock.patch.object(svc, "_build_legacy_agent") as build_legacy:
            req = AgentRequest(message="Марина, поставь бодрое", tools=TOOLS)
            out = await svc.run(req)

        build_tc.assert_called_once_with()
        build_legacy.assert_not_called()
        self.assertIsInstance(out, ToolCallResponse)
        self.assertEqual(out.tool_calls[0].name, "enqueue")
        self.assertEqual(out.tool_calls[0].arguments.tracks[0].id, "a")
        self.assertEqual(out.spoken_answer, "ok")  # derived from display_text

    async def test_no_tools_uses_legacy_agent(self):
        svc = AgentService()
        draft = AgentDraft(display_text="ok 🎵", action="play", track_ids=[])
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


class SystemTextTest(unittest.TestCase):
    """The system prompt (built per request) must embed the bot actions and the
    toolcall agent must build without error."""

    def test_toolcall_system_text_lists_actions(self):
        req = AgentRequest(message="hi", tools=TOOLS)
        text = AgentService()._system_text(req)
        self.assertIn('"play"', text)
        self.assertIn('"pause"', text)

    def test_legacy_system_text_without_tools(self):
        req = AgentRequest(message="hi")
        text = AgentService()._system_text(req)
        self.assertNotIn("Available actions", text)

    def test_build_toolcall_agent(self):
        with mock.patch.dict(os.environ, _ENV):
            agent = AgentService()._build_toolcall_agent()
        self.assertIsNotNone(agent)


if __name__ == "__main__":
    unittest.main()
