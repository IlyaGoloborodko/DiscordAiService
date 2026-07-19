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
    AgentSession,
    ToolCallDraft,
    ToolCallResponse,
    ToolSpec,
    Track,
)
from app.services.agent_service import AgentDeps, AgentService, clean_for_tts

_ENV = {"TM_MODEL_NAME": "dummy", "TM_BASE_URL": "http://127.0.0.1:9/v1", "TM_API_KEY": "x"}

_TRACKS_SCHEMA = {"type": "object", "properties": {"tracks": {"type": "array"}}, "required": ["tracks"]}

TOOLS = [
    ToolSpec(name="play", description="Play now", input_schema=_TRACKS_SCHEMA),
    ToolSpec(name="enqueue", description="Add to queue", input_schema=_TRACKS_SCHEMA),
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
        self.assertEqual(calls[0].arguments["tracks"][0]["id"], "x")

    def test_control_needs_no_tracks(self):
        calls, clean = AgentService._build_tool_calls("pause", [], self.TOOLS)
        self.assertTrue(clean)
        self.assertEqual(calls[0].name, "pause")


class SchemaDrivenArgumentsTests(unittest.TestCase):
    """`arguments` is shaped by the chosen tool's own input_schema, so the bot can add
    a tool with any arguments without a change here. Nothing is hardcoded per tool."""

    SET_VOLUME = ToolSpec(
        name="set_volume",
        description="Set the volume to an exact level on a 1-10 scale.",
        input_schema={
            "type": "object",
            "properties": {"level": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["level"],
        },
    )
    PLAY = ToolSpec(
        name="play",
        input_schema={"type": "object", "properties": {"tracks": {"type": "array"}}, "required": ["tracks"]},
    )
    PAUSE = ToolSpec(name="pause", input_schema={})
    TOOLS = [SET_VOLUME, PLAY, PAUSE]

    def _call(self, action, tracks=(), args_json=""):
        return AgentService._build_tool_calls(action, list(tracks), self.TOOLS, args_json)

    def test_scalar_argument_is_passed_through(self):
        calls, clean = self._call("set_volume", args_json='{"level": 6}')
        self.assertTrue(clean)
        self.assertEqual(calls[0].arguments, {"level": 6})

    def test_argument_is_coerced_to_declared_type(self):
        # The model is loose about types ("1" vs 1); the bot validates against schema.
        calls, clean = self._call("set_volume", args_json='{"level": "1"}')
        self.assertTrue(clean)
        self.assertEqual(calls[0].arguments, {"level": 1})

    def test_missing_required_argument_is_dirty(self):
        calls, clean = self._call("set_volume", args_json="")
        self.assertEqual(calls, [])
        self.assertFalse(clean)

    def test_unparseable_args_are_dirty_not_crashing(self):
        calls, clean = self._call("set_volume", args_json="level 6")
        self.assertEqual(calls, [])
        self.assertFalse(clean)

    def test_track_tool_still_gets_tracks_and_no_stray_args(self):
        calls, clean = self._call("play", tracks=[Track(id="x", title="t")], args_json='{"level": 6}')
        self.assertTrue(clean)
        self.assertEqual(list(calls[0].arguments), ["tracks"])  # `level` is not in play's schema
        self.assertEqual(calls[0].arguments["tracks"][0]["id"], "x")

    def test_argumentless_tool_gets_empty_arguments(self):
        calls, clean = self._call("pause")
        self.assertTrue(clean)
        self.assertEqual(calls[0].arguments, {})

    def test_rendered_tools_expose_the_argument_schema(self):
        rendered = AgentService._render_tools(self.TOOLS)
        self.assertIn("level=<integer 1-10>", rendered)
        self.assertIn("needs tracks", rendered)
        self.assertIn('"pause"', rendered)


class _AgentRunHarness(unittest.IsolatedAsyncioTestCase):
    """Runs AgentService.run with the LLM and memory stubbed out. No test methods of
    its own — subclasses bring those."""

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


class DirtyTurnNotSavedTests(_AgentRunHarness):
    """A turn that produced no valid action (hallucinated / no tracks) is not saved."""

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
        self.assertEqual(resp.tool_calls[0].arguments["tracks"][0]["title"], "Song")

    async def test_invented_id_dropped(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="ok", action="play", track_ids=["ghost"])
        save, resp = await self._run(draft, tools)  # nothing found -> no real tracks
        self.assertEqual(resp.tool_calls, [])
        save.assert_not_called()


class ReplyMatchesToolCallsTests(_AgentRunHarness):
    """The reply must never claim an action the service dropped: the bot speaks it
    aloud and the user believes their ears."""

    async def test_dropped_action_does_not_claim_success(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="Putting on some metal!", action="play")
        _, resp = await self._run(draft, tools)  # no tracks -> action dropped
        self.assertEqual(resp.tool_calls, [])
        self.assertNotIn("metal", resp.display_text.lower())
        self.assertEqual(resp.display_text, AgentService._action_failed_text())
        self.assertEqual(resp.spoken_answer, clean_for_tts(resp.display_text))

    async def test_hallucinated_action_does_not_claim_success(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="Playing it now!", action="play_track")
        _, resp = await self._run(draft, tools)
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.display_text, AgentService._action_failed_text())

    async def test_delivered_action_keeps_the_models_reply(self):
        tools = [ToolSpec(name="play", input_schema={"required": ["tracks"]})]
        draft = ToolCallDraft(display_text="Putting on some metal!", action="play", track_ids=["a"])
        _, resp = await self._run(draft, tools, found={"a": Track(id="a", title="Song")})
        self.assertEqual(resp.display_text, "Putting on some metal!")
        self.assertEqual(resp.tool_calls[0].name, "play")


class DurationFilterTests(unittest.TestCase):
    """Hour-long compilations are dropped before the model can pick them: the bot
    treats each item as one track, so a mix is one un-skippable queue entry."""

    def test_long_mix_dropped_short_track_kept(self):
        deps = AgentDeps(search=None)
        song = Track(id="s", title="Song", duration=303.0)
        mix = Track(id="m", title="Thrash Metal Mix", duration=6493.0)
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([song, mix])
        self.assertEqual([t.id for t in kept], ["s"])
        self.assertNotIn("m", deps.found)  # model cannot resolve it either

    def test_unknown_duration_kept(self):
        deps = AgentDeps(search=None)
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([Track(id="x", title="Chart hit", duration=None)])
        self.assertEqual([t.id for t in kept], ["x"])

    def test_limit_is_configurable(self):
        deps = AgentDeps(search=None)
        mix = Track(id="m", title="Long", duration=1200.0)
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "1800"}):
            self.assertEqual([t.id for t in deps.remember([mix])], ["m"])


class SilentCommandTests(_AgentRunHarness):
    """Controls are carried out without a spoken line: you hear pause/skip/volume
    happen, so announcing them first only delays the thing that was asked for."""

    VOLUME = ToolSpec(
        name="set_volume",
        input_schema={"type": "object", "properties": {"level": {"type": "integer"}}, "required": ["level"]},
    )
    PAUSE = ToolSpec(name="pause", input_schema={})
    PLAY = ToolSpec(
        name="play",
        input_schema={"type": "object", "properties": {"tracks": {"type": "array"}}, "required": ["tracks"]},
    )
    TOOLS = [VOLUME, PAUSE, PLAY]

    async def test_pause_is_not_spoken(self):
        draft = ToolCallDraft(display_text="Ставлю на паузу", action="pause")
        _, resp = await self._run(draft, self.TOOLS)
        self.assertEqual(resp.tool_calls[0].name, "pause")
        self.assertEqual(resp.spoken_answer, "")
        self.assertEqual(resp.display_text, "Ставлю на паузу")  # chat still gets it

    async def test_volume_is_not_spoken(self):
        draft = ToolCallDraft(display_text="Делаю громче", action="set_volume", action_args_json='{"level": 6}')
        _, resp = await self._run(draft, self.TOOLS)
        self.assertEqual(resp.tool_calls[0].arguments, {"level": 6})
        self.assertEqual(resp.spoken_answer, "")

    async def test_playing_music_is_still_spoken(self):
        draft = ToolCallDraft(display_text="Включаю метал", action="play", track_ids=["a"])
        _, resp = await self._run(draft, self.TOOLS, found={"a": Track(id="a", title="Song")})
        self.assertEqual(resp.spoken_answer, "Включаю метал")

    async def test_answering_a_question_is_still_spoken(self):
        # No tool call at all — this is conversation, and must be heard.
        draft = ToolCallDraft(display_text="Сейчас громкость 5")
        _, resp = await self._run(draft, self.TOOLS)
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.spoken_answer, "Сейчас громкость 5")


class TriggerTests(unittest.TestCase):
    """Only a person may ask to hear the last hour again. When the bot starts the
    turn itself (queue ran out, DJ break) the job is to find something new."""

    @staticmethod
    def _asked(**kwargs):
        from app.services.agent_service import _asked_by_a_person

        session = AgentSession(**{k: v for k, v in kwargs.items() if k != "trigger"})
        return _asked_by_a_person(
            AgentRequest(message="x", session=session, trigger=kwargs.get("trigger"))
        )

    def test_explicit_user_trigger(self):
        self.assertTrue(self._asked(trigger="user"))

    def test_autoplay_and_dj_break_are_machine_turns(self):
        self.assertFalse(self._asked(trigger="autoplay"))
        self.assertFalse(self._asked(trigger="dj_break"))

    def test_any_unknown_trigger_counts_as_machine(self):
        # The bot may add new automatic reasons; anything but "user" is one.
        self.assertFalse(self._asked(trigger="queue_stalled"))

    def test_trigger_beats_the_guess(self):
        # A machine turn that still names a user must stay a machine turn — this is
        # exactly the silent failure the explicit field was added to prevent.
        self.assertFalse(self._asked(trigger="autoplay", user_id="42", user_name="Sok"))

    def test_falls_back_to_guessing_without_a_trigger(self):
        self.assertTrue(self._asked(user_id="42"))
        self.assertFalse(self._asked())


class RestingFilterTests(unittest.TestCase):
    """Tracks that played recently are hidden from the model — unless hiding them
    would leave it with nothing to offer."""

    SONG = Track(id="a", title="Song", duration=200.0)
    OTHER = Track(id="b", title="Other", duration=210.0)
    MIX = Track(id="m", title="1 HOUR PHONK MIX", duration=3600.0)

    def test_resting_track_is_hidden(self):
        deps = AgentDeps(search=None, resting={"a"})
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([self.SONG, self.OTHER])
        self.assertEqual([t.id for t in kept], ["b"])
        self.assertNotIn("a", deps.found)

    def test_all_resting_falls_back_to_repeating(self):
        # Real case: a "phonk" search is nearly all hour-long mixes, so once the
        # few real tracks have played there is nothing left. Repeating beats
        # telling the user we found nothing.
        deps = AgentDeps(search=None, resting={"a", "b"})
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([self.SONG, self.OTHER])
        self.assertEqual([t.id for t in kept], ["a", "b"])

    def test_fallback_never_lets_long_mixes_through(self):
        # The length filter is a hard rule: an hour-long mix breaks the queue.
        deps = AgentDeps(search=None, resting={"a"})
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([self.SONG, self.MIX])
        self.assertEqual([t.id for t in kept], ["a"])  # fell back, but no mix

    def test_recently_played_tool_ignores_resting(self):
        deps = AgentDeps(search=None, resting={"a"})
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([self.SONG], include_resting=True)
        self.assertEqual([t.id for t in kept], ["a"])

    def test_machine_turn_cannot_replay_the_recent_tracks(self):
        # Real bug: the queue ran out, the model reached for "what were we playing"
        # and re-served the whole previous hour, cooldown and all.
        deps = AgentDeps(search=None, resting={"a"}, may_replay_recent=False)
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            kept = deps.remember([self.SONG, self.OTHER], include_resting=deps.may_replay_recent)
        self.assertEqual([t.id for t in kept], ["b"])

    def test_source_query_is_recorded(self):
        deps = AgentDeps(search=None)
        with mock.patch.dict(os.environ, {"MAX_TRACK_SECONDS": "600"}):
            deps.remember([self.SONG], "phonk")
        self.assertEqual(deps.source_queries["a"], "phonk")


class ChartsFallbackTests(unittest.IsolatedAsyncioTestCase):
    """A non-English tag returns nothing from the chart source; the tool must fall
    back to a plain search rather than hand the model an empty list."""

    @staticmethod
    def _charts_tool(agent):
        # The tools are registered as closures; grab the one under test by name.
        return agent._function_toolset.tools["get_top_charts"].function

    async def _call(self, charts_result, search_result, tag="металл"):
        with mock.patch.dict(os.environ, _ENV):
            agent = AgentService()._build_toolcall_agent()
        search = mock.Mock()
        search.charts = mock.AsyncMock(return_value=charts_result)
        search.search = mock.AsyncMock(return_value=search_result)
        deps = AgentDeps(search=search)
        ctx = mock.Mock(deps=deps)
        tracks = await self._charts_tool(agent)(ctx, tag=tag)
        return tracks, search

    async def test_empty_charts_falls_back_to_search(self):
        hit = Track(id="a", title="Slipknot - Psychosocial", duration=303.0)
        tracks, search = await self._call([], [hit])
        search.search.assert_awaited_once()
        self.assertEqual([t.id for t in tracks], ["a"])

    async def test_non_empty_charts_does_not_search(self):
        hit = Track(id="c", title="Chart hit", duration=200.0)
        tracks, search = await self._call([hit], [])
        search.search.assert_not_awaited()
        self.assertEqual([t.id for t in tracks], ["c"])

    async def test_no_tag_and_empty_charts_does_not_search(self):
        # Without a tag there is nothing to search for; the global top is legitimately
        # empty-able and must not turn into a bogus search.
        _, search = await self._call([], [], tag="")
        search.search.assert_not_awaited()


class RecentTracksTests(unittest.TestCase):
    """Recently played tracks are a short, de-duplicated tail — not a play history."""

    @staticmethod
    def _merge(played, existing, limit=3):
        from app.services.memory import MemoryStore

        return [t["id"] for t in MemoryStore._merge_recent(played, existing, limit)]

    def test_newest_first_and_capped(self):
        merged = self._merge([{"id": "a"}], [{"id": "b"}, {"id": "c"}, {"id": "d"}])
        self.assertEqual(merged, ["a", "b", "c"])  # "d" falls off the tail

    def test_replayed_track_is_not_duplicated(self):
        merged = self._merge([{"id": "b"}], [{"id": "a"}, {"id": "b"}])
        self.assertEqual(merged, ["b", "a"])

    def test_tracks_without_id_are_skipped(self):
        # An id-less track cannot be replayed, so keeping it would waste the cap.
        self.assertEqual(self._merge([{"title": "no id"}, {"id": "a"}], []), ["a"])

    def test_limit_is_configurable(self):
        with mock.patch.dict(os.environ, {"RECENT_TRACKS_LIMIT": "7"}):
            from app.services.memory import MemoryStore

            self.assertEqual(MemoryStore().track_limit, 7)


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
    def test_every_context_field_reaches_the_prompt(self):
        # Regression: the field list was hardcoded, so `volume` never reached the
        # model and it truthfully answered "I can't see the volume".
        req = AgentRequest(
            message="какая громкость?",
            context={
                "now_playing": "SOAD - Chop Suey!",
                "queue": ["Deftones - Change"],
                "queue_len": 1,
                "volume": 5,
            },
        )
        prompt = AgentService._format_prompt(req)
        self.assertIn("Volume (1-10 scale): 5", prompt)
        self.assertIn("Now playing: SOAD - Chop Suey!", prompt)
        self.assertIn("Queue: Deftones - Change", prompt)

    def test_unknown_context_field_is_still_rendered(self):
        # A field the bot adds later must arrive without a change here.
        req = AgentRequest(message="hi", context={"repeat_mode": "all"})
        self.assertIn("Repeat mode: all", AgentService._format_prompt(req))

    def test_empty_context_values_are_skipped(self):
        req = AgentRequest(message="hi", context={"now_playing": None, "queue": [], "volume": 5})
        prompt = AgentService._format_prompt(req)
        self.assertIn("Volume (1-10 scale): 5", prompt)
        self.assertNotIn("Now playing", prompt)
        self.assertNotIn("Queue:", prompt)

    def test_render_tools_lists_names_as_strings(self):
        rendered = AgentService._render_tools(TOOLS)
        # Names are quoted string values, not function-signature-looking entries.
        self.assertIn('"play" (needs tracks)', rendered)
        self.assertIn('"pause"', rendered)
        self.assertNotIn("arguments schema", rendered)

    def test_format_prompt_includes_now_playing_and_queue(self):
        req = AgentRequest(
            message="Бот, что в очереди?",
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
            req = AgentRequest(message="Бот, поставь бодрое", tools=TOOLS)
            out = await svc.run(req)

        build_tc.assert_called_once_with()
        build_legacy.assert_not_called()
        self.assertIsInstance(out, ToolCallResponse)
        self.assertEqual(out.tool_calls[0].name, "enqueue")
        self.assertEqual(out.tool_calls[0].arguments["tracks"][0]["id"], "a")
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
