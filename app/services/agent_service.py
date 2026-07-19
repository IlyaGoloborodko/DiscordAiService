import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import emoji

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Safety net: retry a run when the LM Studio engine returns a 400 rejecting its own
# output (a stochastic tool-call/parser failure). qwen3.5-9b almost never trips this,
# but pydantic-ai's own `retries` only covers output validation, not HTTP errors.
_MODEL_HTTP_RETRIES = 3

from app.data.models import (
    AgentDraft,
    AgentRequest,
    AgentResponse,
    AgentSession,
    ToolCall,
    ToolCallDraft,
    ToolCallResponse,
    ToolSpec,
    Track,
)
from app.recommendations import cooldown, genres, sampling
from app.recommendations import history as play_history
from app.services.memory import MemoryStore
from app.services.search_client import SearchClient

logger = logging.getLogger(__name__)

# Shared persona + search-tool guidance, reused by both response modes.
_PERSONA = """\
You are {NAME}, a voice DJ assistant for a Discord music bot. The user talks to \
you (often in Russian). The message may start by addressing you ({NAME_FORMS}) — \
that is just how they call you, not part of the command; ignore it as content.

*** LANGUAGE RULE — HIGHEST PRIORITY, NO EXCEPTIONS ***
`display_text` MUST be written ENTIRELY in {LANG}. NEVER write in any other language, \
even when the user writes to you in a different language (e.g. Russian). This is a \
hard technical constraint: `display_text` is read aloud by a {LANG} text-to-speech \
voice, and text in any other language comes out as garbled, unintelligible noise. \
Track and artist names may keep their original spelling, but every word YOU write \
must be {LANG}. If you are about to write a word in another language, translate it \
to {LANG} instead.

Write ONE reply in `display_text`. It is shown in the chat as-is AND, after the \
service strips emoji/markdown, is read aloud by that {LANG} TTS voice. Keep it short, \
natural and conversational; a few emoji are fine (they are removed before speech). Do \
NOT produce a separate spoken field — the service derives it from `display_text`.

Always use the tools to find real tracks before recommending anything — never \
invent track ids. Pick the most fitting tool:
- `search_tracks(query)`: an explicitly named track or artist, or plain free-text search.
- `get_playlist_tracks(url)`: the user gives a playlist link or asks for a specific playlist.
- `get_similar_tracks(artist, track)`: "something like X" / "in the style of X" — \
pass the artist AND a representative seed track; both are needed.
- `get_recently_played()`: the user refers to what was already on ("продолжи прошлый \
плейлист", "верни то же", "что мы слушали"). Prefer it over searching — it returns the \
actual earlier tracks. If it comes back empty, say so instead of inventing a playlist.

When the user asks for MORE of something already playing, or explicitly for something \
NEW / different / "не повторяй" — call `get_recently_played()` FIRST, then pick only \
ids that are NOT in that list. Searching the same genre twice returns the same popular \
hits, so without checking you will hand back tracks they just heard. If everything you \
found was already played, search again with a narrower or different query (a sub-genre, \
another artist) rather than repeating.
- `get_top_charts(tag, country)`: "what's popular / trending" or "an hour of \
<genre>/<mood>" — `tag` for a genre or mood, `country` when named; omit both for the global top.

`tag` MUST be an ENGLISH genre/mood word — the chart source only indexes English \
tags. Translate the user's word first: "металл" -> "metal", "фонк" -> "phonk", \
"тяжёлое" -> "metal". Never pass Russian to `tag`. And never fall back to calling \
`get_top_charts` with NO tag when a genre was asked for — that returns the global \
pop chart, not what the user wanted; use `search_tracks(<genre>)` instead.
"""

# Both modes: the model returns display_text + a single `action` + a FLAT list of
# `track_ids` (strings). Never a nested tracks array — local models can't emit that.
LEGACY_PROMPT = _PERSONA + """
Set `action`:
- "play": user wants one track / to start now -> put the best track's id in `track_ids`.
- "enqueue": user wants several tracks / a playlist / "an hour of music" -> put several ids in `track_ids`.
- "replace_queue": user wants to replace what's playing with a new set.
- "clarify": too vague to act on -> ask in `clarification`, `track_ids` empty.
- "none": small talk -> just answer, `track_ids` empty.

Charts and recommendations usually mean several tracks -> prefer "enqueue".
`track_ids` must contain ONLY ids returned by the search tools — copy them exactly, never invent.
"""

# Tool-calling mode header; the concrete bot actions are appended per request.
_TOOLCALL_RULES = """
The ONLY functions you may call are the search tools (search_tracks,
get_playlist_tracks, get_similar_tracks, get_top_charts) and `final_result`. You
ALWAYS answer by calling `final_result`, and never call anything else.

You do NOT call the bot actions. Instead set the `action` field of `final_result`
to ONE action name from the list below (a plain string), or "" for no action.

MANDATORY: if the user wants music (put on / start / play / add / a genre / a mood /
an artist / "your choice"), FIRST call a search tool to get real tracks, THEN call
`final_result` with action "play" or "enqueue" and `track_ids` set to the ids you
just got. Never describe music in display_text without setting an action + track_ids.
If unsure of the genre, still pick a popular selection and offer to refine next time.

Some actions take arguments of their own — they are listed after the action name
below. Put those in `action_args_json` as a flat JSON object using exactly the listed
names, e.g. an action listed as `level=<integer 1-10>` -> `action_args_json` =
{"level": 6}. Numbers written as words are still numbers ("до единицы" -> 1). Leave
`action_args_json` as "" for actions that take no arguments or only take tracks.
`track_ids` is never put in `action_args_json` — it has its own field.

Rules:
- Controls (pause/resume/skip/stop/...): set `action` to that name, `track_ids` empty.
- The "Current player state" block in the user message is the AUTHORITATIVE live state
  of the player. Questions about it ("какая громкость", "что играет", "что дальше")
  are answered directly from those values, with action "" and no tools — the tools
  only CHANGE state, they cannot read it. Quote the value you were given. If a field
  you were asked about is absent from that block, say you don't have it — never guess
  a number.
- Small talk / nothing to do: action "", track_ids empty.
- `track_ids` must be ids returned by the search tools — copy them exactly, never invent.

Allowed action names (plain strings for the `action` field):
"""


# TTS (Piper) chokes on emoji/markdown and falls back to reading them out
# character-by-character. spoken_answer must be plain text, so we strip those
# server-side rather than trust the model to obey the prompt. Emoji removal uses
# the `emoji` library (full, maintained Unicode emoji data) instead of hand-rolled
# ranges, which kept missing symbols (™, ‼, CJK marks, ...).
_MARKDOWN_RE = re.compile(r"[*_`~#>|]+")
# A few decorative symbols that are NOT emoji per the Unicode spec (so the emoji
# library leaves them) but that TTS mispronounces. Kept as an explicit tiny set,
# not ranges — musical notes are likely in a music bot's replies.
_EXTRA_SYMBOLS_RE = re.compile("[♪♫♬♩★☆]")


def clean_for_tts(text: str) -> str:
    """Strip emoji and markdown so the spoken text stays coherent for TTS."""
    text = emoji.replace_emoji(text, "")
    text = _EXTRA_SYMBOLS_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# Bump when the response contract changes so stale history from the old contract
# is abandoned instead of poisoning the model as few-shot examples. v2 abandoned the
# pre-tool English legacy answers; v3 abandons histories poisoned by hallucinated
# action names ("play_track"); v4 abandons histories where the assistant emitted a
# `spoken_answer` field (removed from the schema — the service derives it now), which
# the model copied from history and which broke the gemma tool-call parser; v5
# abandons the nested tool_calls-shaped history after the output schema was flattened
# to a single action + flat track_ids; v6 abandons gemma-4-e4b history after switching
# the model to qwen3.5-9b (different tool-call template — don't cross-poison); v7
# abandons the raw agentic transcript (tool calls + tool returns) that we used to
# persist — we now store only clean user/assistant-text turns (see run()).
_MEMORY_VERSION = "v7"


# The search backend interleaves real tracks with hour-long compilations ("Thrash
# Metal Mix", 108 min). The bot treats every item as ONE track, so a mix becomes a
# single un-skippable queue entry, `skip` throws away an hour, and the periodic DJ
# break never fires. Drop them here — the model never even sees them.
_DEFAULT_MAX_TRACK_SECONDS = 600


def _max_track_seconds() -> int:
    try:
        return int(os.getenv("MAX_TRACK_SECONDS") or _DEFAULT_MAX_TRACK_SECONDS)
    except ValueError:
        return _DEFAULT_MAX_TRACK_SECONDS


# Nicer labels for the player-state fields the bot is known to send. Anything not
# listed is still rendered (from its own key), so a new field needs no change here.
_CONTEXT_LABELS = {
    "now_playing": "Now playing",
    "queue": "Queue",
    "queue_len": "Queue length",
    "volume": "Volume (1-10 scale)",
}

# Spoken when the model promised an action the service had to drop. Must be in
# BOT_LANGUAGE; override via TEXT_ACTION_FAILED when the bot speaks another language.
_DEFAULT_ACTION_FAILED = (
    "Sorry, I couldn't find anything that fits — want to try another genre or artist?"
)


def _is_silent_command(tool_calls: list[ToolCall]) -> bool:
    """Whether this turn is a plain control the bot should just do, without talking.

    Pause, resume, skip, stop and the volume actions all produce an instant,
    obvious result — the listener hears it happen. Announcing it first only adds
    the text-to-speech delay in front of the thing they asked for.

    A command is anything that acts without tracks. Deciding it that way rather
    than by listing names means a control the bot adds later is silent too,
    without a change here. Answering a question ("what's the volume?") makes no
    tool call at all, so it still gets spoken.
    """
    return bool(tool_calls) and all(not call.arguments.get("tracks") for call in tool_calls)


def _coerce_to_schema(value: Any, prop: dict[str, Any]) -> Any:
    """Coerce a model-supplied argument to the JSON-schema type the bot declared.
    The model reliably picks the right value but is loose about its type ("6" vs 6),
    and the bot validates against its schema. Returns None when the value cannot be
    represented — the caller then treats the argument as missing."""
    declared = prop.get("type")
    try:
        if declared == "integer":
            return int(float(value)) if not isinstance(value, bool) else None
        if declared == "number":
            return float(value)
        if declared == "boolean":
            return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
        if declared == "string":
            return value if isinstance(value, str) else str(value)
    except (TypeError, ValueError):
        return None
    return value  # no declared type (or a compound one): pass through untouched


def fits_duration(track: Track) -> bool:
    """Whether a search hit is a real track rather than a long mix. Unknown duration
    is kept: charts/similar results often omit it and are usually real tracks."""
    return track.duration is None or track.duration <= _max_track_seconds()


@dataclass
class AgentDeps:
    search: SearchClient
    # Tracks returned by the search tools this run, keyed by id. The model outputs
    # only ids; we resolve them back to full tracks here (and drop invented ids).
    found: dict[str, Track] = field(default_factory=dict)
    # What this session played recently (newest first), loaded from memory. Lets the
    # model answer "continue the previous playlist" — the Discord bot forgets its
    # queue on restart, so the service is the only place this survives.
    recent: list[Track] = field(default_factory=list)
    # Tracks that played recently enough to still be resting (see recommendations
    # .cooldown). Hidden from the model so it cannot pick them at all.
    resting: set[str] = field(default_factory=set)
    # Whether "give me what we were listening to" is allowed to serve tracks that
    # are still resting. True when a person asked for it in words; False when the
    # turn was triggered by the bot itself (the queue ran dry, a DJ break), where
    # the job is to find something NEW and replaying the last hour is a bug.
    may_replay_recent: bool = True
    # Which search query produced each track, so the history can record a rough
    # genre for it ("metal", "phonk") until real tags exist.
    source_queries: dict[str, str] = field(default_factory=dict)

    def remember(
        self, tracks: list[Track], query: str = "", include_resting: bool = False
    ) -> list[Track]:
        """Cache what a search tool found and return what the model may use.

        Two kinds of track never make it through: hour-long compilations, and
        tracks that are still resting after playing recently. `include_resting`
        is for the one tool that deliberately returns recent tracks.
        """
        real_tracks = [track for track in tracks if fits_duration(track)]
        keep = (
            real_tracks
            if include_resting
            else [track for track in real_tracks if track.id not in self.resting]
        )
        # Some genres are almost entirely hour-long mixes — a search for "phonk"
        # comes back with 25 results of which 4 are actual songs. Once those four
        # have played, resting them all would leave nothing at all, and the user
        # would be told we found nothing. Hearing a track again is better than
        # that, so when the rest empties the list we ignore it. The length filter
        # is never relaxed: an hour-long mix breaks the player's queue.
        if not keep and real_tracks:
            logger.info("all %d candidates were resting; letting them through", len(real_tracks))
            keep = real_tracks
        for track in keep:
            self.found[track.id] = track
            if query and track.id not in self.source_queries:
                self.source_queries[track.id] = query
        return keep


class AgentService:
    """Builds and runs the music agent. Dual-mode: when the request declares
    `tools`, the agent returns bot tool_calls; otherwise it returns the legacy
    action + tracks shape. The model/provider are read from env at request time
    (env is loaded after import), matching the existing services."""

    def __init__(self) -> None:
        self.memory = MemoryStore()

    # --- agent construction --------------------------------------------------

    def _model(self) -> OpenAIChatModel:
        return OpenAIChatModel(
            model_name=os.getenv("TM_MODEL_NAME"),
            provider=OpenAIProvider(
                base_url=os.getenv("TM_BASE_URL"),
                api_key=os.getenv("TM_API_KEY"),
            ),
        )

    @staticmethod
    def _register_search_tools(agent: Agent[AgentDeps, object]) -> None:
        @agent.tool
        async def search_tracks(
            ctx: RunContext[AgentDeps], query: str, limit: int = 10
        ) -> list[Track]:
            """Search for tracks by a free-text query (artist, track, genre, mood)."""
            found = await ctx.deps.search.search(query, sampling.pool_size(limit))
            return sampling.pick_varied(ctx.deps.remember(found, query), limit)

        @agent.tool
        async def get_playlist_tracks(
            ctx: RunContext[AgentDeps], url: str, limit: int = 50
        ) -> list[Track]:
            """Expand a YouTube playlist (URL or bare playlist id) into its tracks."""
            # A playlist is an explicit, ordered thing the user asked for — take it
            # as it comes instead of sampling it.
            return ctx.deps.remember(await ctx.deps.search.playlist(url, limit), url)

        @agent.tool
        async def get_similar_tracks(
            ctx: RunContext[AgentDeps],
            artist: str,
            track: str = "",
            limit: int = 10,
        ) -> list[Track]:
            """Recommend tracks in the style of an artist. Always pass a seed
            `track` too (a well-known song by that artist) — the recommendation
            source needs both."""
            found = await ctx.deps.search.similar(
                artist, track or None, sampling.pool_size(limit)
            )
            return sampling.pick_varied(ctx.deps.remember(found, artist), limit)

        @agent.tool
        async def get_recently_played(ctx: RunContext[AgentDeps], limit: int = 20) -> list[Track]:
            """Tracks this session played recently, newest first. Use for "continue
            the previous playlist", "put that back on", "what were we listening to" —
            these refer to real earlier tracks, so return them instead of searching."""
            # These tracks are recent by definition, so they are all resting. When
            # a person actually asked for them back, that is the point and the rest
            # is waived. When the bot triggered this turn itself, it is not: the
            # model reaches for this tool on "play the next set" too, and replaying
            # the last hour is exactly what we are trying to avoid.
            return ctx.deps.remember(
                ctx.deps.recent[:limit], include_resting=ctx.deps.may_replay_recent
            )

        @agent.tool
        async def get_top_charts(
            ctx: RunContext[AgentDeps],
            tag: str = "",
            country: str = "",
            limit: int = 10,
        ) -> list[Track]:
            """Popular/trending tracks. `tag` is a genre or mood and MUST be an
            English word — the chart source only indexes English tags, so translate
            the user's genre first ("металл" -> "metal"). `country` when the user
            names one; omit both for the global top."""
            pool = sampling.pool_size(limit)
            tracks = await ctx.deps.search.charts(tag or None, country or None, pool)
            if not tracks and tag:
                # The chart source only knows English tags, so a non-English or
                # obscure one returns nothing. Fall back to a plain search instead of
                # handing back an empty list: on empty the model either gives up
                # (action set, no track ids) or silently retries with no tag at all,
                # which is the global top — pop/hip-hop regardless of what was asked.
                tracks = await ctx.deps.search.search(tag, pool)
            return sampling.pick_varied(ctx.deps.remember(tracks, tag or country), limit)

    # The system prompt is injected fresh into message_history each run (see run),
    # not set on the Agent — so it can never be lost when memory trims history and
    # is always the current version.
    # Low temperature for reliable tool use; max_tokens caps the reply (pydantic-ai
    # maps it to max_completion_tokens). `openai_reasoning_effort='none'` is REQUIRED
    # for GPT-5.x reasoning models to use function tools over /v1/chat/completions —
    # otherwise OpenAI 400s ("... not supported ... set reasoning_effort to 'none'").
    # It also turns the reasoning preamble off, which is what we want: this is fast
    # action routing, not a task that benefits from chain-of-thought. Harmless on
    # non-reasoning models. Bump/relax per model if a future one needs to think.
    _MODEL_SETTINGS = {"temperature": 0.2, "max_tokens": 2048, "openai_reasoning_effort": "none"}

    def _build_legacy_agent(self) -> Agent[AgentDeps, AgentDraft]:
        agent = Agent(
            self._model(),
            output_type=AgentDraft,
            deps_type=AgentDeps,
            retries=5,
            model_settings=self._MODEL_SETTINGS,
        )
        self._register_search_tools(agent)
        return agent

    def _build_toolcall_agent(self) -> Agent[AgentDeps, ToolCallDraft]:
        agent = Agent(
            self._model(),
            output_type=ToolCallDraft,
            deps_type=AgentDeps,
            retries=5,
            model_settings=self._MODEL_SETTINGS,
        )
        self._register_search_tools(agent)
        return agent

    def _system_text(self, request: AgentRequest) -> str:
        if request.tools:
            text = _PERSONA + _TOOLCALL_RULES + self._render_tools(request.tools)
        else:
            text = LEGACY_PROMPT
        # Bot identity + language, configured via env so they aren't hardcoded in
        # the prompt. NAME_FORMS lists the ways the user addresses the bot (name +
        # its inflected/nickname forms); the TTS voice must match BOT_LANGUAGE.
        name = self._bot_name()
        return (
            text.replace("{NAME_FORMS}", self._bot_name_forms(name))
            .replace("{NAME}", name)
            .replace("{LANG}", self._bot_language())
        )

    @staticmethod
    def _bot_name() -> str:
        return (os.getenv("BOT_NAME") or "Marina").strip() or "Marina"

    @staticmethod
    def _bot_name_forms(name: str) -> str:
        raw = os.getenv("BOT_NAME_FORMS") or ""
        forms = [f.strip() for f in raw.split(",") if f.strip()]
        return ", ".join(forms) if forms else name

    @staticmethod
    def _bot_language() -> str:
        return (os.getenv("BOT_LANGUAGE") or "English").strip() or "English"

    @staticmethod
    def _action_failed_text() -> str:
        """Reply sent when the model claimed an action the service could not deliver
        (unknown action name, or a music action with no real tracks). Env-configurable
        because it is spoken aloud and so must be written in BOT_LANGUAGE."""
        return (os.getenv("TEXT_ACTION_FAILED") or "").strip() or _DEFAULT_ACTION_FAILED

    @classmethod
    def _render_tools(cls, tools: list[ToolSpec]) -> str:
        # List names as plain strings + what they mean. Deliberately NOT shaped like
        # function signatures, so the model doesn't try to call them natively. Each
        # action's own arguments are rendered from its input_schema, so a bot can add
        # a tool taking any arguments without a change here.
        lines = []
        for tool in tools:
            lines.append(f'- "{tool.name}"{cls._render_tool_args(tool)}: {tool.description}')
        return "\n".join(lines)

    @staticmethod
    def _render_tool_args(tool: ToolSpec) -> str:
        schema = tool.input_schema or {}
        properties: dict[str, Any] = schema.get("properties") or {}
        required = schema.get("required") or []
        names = list(properties) or list(required)

        hints: list[str] = []
        for name in names:
            if name == "tracks":
                hints.append("needs tracks")
                continue
            prop = properties.get(name) or {}
            declared = prop.get("type", "value") if isinstance(prop, dict) else "value"
            bounds = ""
            if isinstance(prop, dict) and "minimum" in prop and "maximum" in prop:
                bounds = f" {prop['minimum']}-{prop['maximum']}"
            hints.append(f"{name}=<{declared}{bounds}> in action_args_json")
        return f" ({'; '.join(hints)})" if hints else ""

    # --- run -----------------------------------------------------------------

    async def run(self, request: AgentRequest) -> AgentResponse | ToolCallResponse:
        session_key = self._session_key(request)
        intermediate = await self.memory.load(session_key)
        recent = [Track(**data) for data in await self.memory.load_tracks(session_key)]
        # Tracks this server heard recently enough that they should sit this one
        # out. Loaded once per run and hidden from the model in `remember`.
        guild_id = request.session.guild_id or ""
        resting = cooldown.resting_track_ids(await play_history.load_play_stats(guild_id))
        # The bot sends no user when it triggers a turn on its own — the queue ran
        # out, or it is time for a DJ break. Nobody asked to hear anything again,
        # so replaying what just played is not allowed on those turns.
        asked_by_a_person = bool(request.session.user_id or request.session.user_name)
        deps = AgentDeps(
            search=SearchClient(),
            recent=recent,
            resting=resting,
            may_replay_recent=asked_by_a_person,
        )

        if request.tools:
            agent: Agent[AgentDeps, object] = self._build_toolcall_agent()
        else:
            agent = self._build_legacy_agent()

        # Prepend the current system prompt so it is always present, regardless of
        # how the intermediate history was trimmed. pydantic-ai does not re-add a
        # system prompt when message_history is non-empty, so we own it here.
        system = ModelRequest(parts=[SystemPromptPart(content=self._system_text(request))])
        history = [system, *intermediate]

        # LM Studio can stochastically return a 400 ModelHTTPError (tool-call parser
        # rejecting the engine's own output), which pydantic-ai does NOT retry — its
        # `retries` only covers output validation. A fresh attempt usually succeeds,
        # so retry the whole run a few times.
        prompt = self._format_prompt(request)
        result = None
        for attempt in range(_MODEL_HTTP_RETRIES):
            try:
                result = await agent.run(prompt, deps=deps, message_history=history)
                break
            except ModelHTTPError:
                if attempt == _MODEL_HTTP_RETRIES - 1:
                    raise
                deps.found.clear()

        draft = result.output
        # Map the model's ids back to full tracks it actually found; invented ids
        # (not in deps.found) are silently dropped.
        tracks = [deps.found[i] for i in draft.track_ids if i in deps.found]

        clean = True
        if isinstance(draft, ToolCallDraft) and request.tools is not None:
            tool_calls, clean = self._build_tool_calls(
                draft.action, tracks, request.tools, draft.action_args_json
            )
            # The reply and the tool_calls must agree. When the action was dropped the
            # model's text still claims it happened ("Putting on some metal") — the bot
            # would speak that and play nothing, and the user believes their ears. So
            # replace the text rather than let it lie.
            text = draft.display_text if clean else self._action_failed_text()
            # Pause, skip and volume speak for themselves — you hear the result
            # immediately, so reading a line out first is just a delay before it.
            # The chat message still goes out; only the voice is skipped.
            spoken = "" if _is_silent_command(tool_calls) else clean_for_tts(text)
            response: AgentResponse | ToolCallResponse = ToolCallResponse(
                spoken_answer=spoken,
                display_text=text,
                clarification=draft.clarification,
                tool_calls=tool_calls,
            )
        else:
            response = AgentResponse(
                spoken_answer=clean_for_tts(draft.display_text),
                display_text=draft.display_text,
                action=draft.action,
                tracks=tracks,
                clarification=draft.clarification,
            )

        # Persist only a CLEAN conversational turn: the user's message and Marina's
        # reply text — never the raw agentic transcript (tool calls, tool returns,
        # reasoning). Replaying that scaffolding is fragile (strict chat templates
        # like qwen's reject a history that doesn't start with the system message or
        # has an orphaned tool-return) and bloats context with search-result JSON.
        # The model gets fresh tools every run; it needs the conversation, not the
        # mechanics. Only clean turns are saved, so a bad answer can't poison the next.
        if clean:
            turn = [
                ModelRequest(parts=[UserPromptPart(content=self._history_user_text(request))]),
                ModelResponse(parts=[TextPart(content=draft.display_text)]),
            ]
            await self.memory.save(session_key, [*intermediate, *turn])
            # Remember what actually went to the player, so a later "continue the
            # previous playlist" has real tracks to return.
            if tracks:
                await self.memory.save_tracks(
                    session_key, [track.model_dump() for track in tracks]
                )
                # ...and write it to the durable listening history, which is what
                # decides how long each track rests before it can play again.
                await play_history.record_plays(
                    guild_id=guild_id,
                    user_id=request.session.user_id,
                    tracks=tracks,
                    action=draft.action or "",
                    source_queries=deps.source_queries,
                )
                # Look up the genres of what just played, but don't make the user
                # wait for it (~0.8s per new artist). Whatever this misses is
                # fetched later, when the taste profile actually needs it.
                genres.warm_cache_in_background(
                    [track.uploader for track in tracks if track.uploader]
                )

        return response

    @staticmethod
    def _history_user_text(request: AgentRequest) -> str:
        """The user turn as stored in memory — who + message, without the volatile
        player-state block (that is fed fresh via `context` each run)."""
        who = request.session.user_name or "Unknown user"
        return f"{who}: {request.message}"

    @classmethod
    def _build_tool_calls(
        cls,
        action: str,
        tracks: list[Track],
        tools: list[ToolSpec],
        action_args_json: str = "",
    ) -> tuple[list[ToolCall], bool]:
        """Turn the model's single (action, tracks, args) into the bot's tool_calls.
        Returns (tool_calls, clean). A hallucinated action name, or an action missing
        a required argument, yields no call and clean=False (won't be saved)."""
        if not action:
            return [], True
        spec = next((t for t in tools if t.name == action), None)
        if spec is None:
            return [], False  # hallucinated action name
        arguments, clean = cls._build_arguments(spec, tracks, action_args_json)
        if not clean:
            return [], False
        return [ToolCall(name=action, arguments=arguments)], True

    @classmethod
    def _build_arguments(
        cls, spec: ToolSpec, tracks: list[Track], action_args_json: str
    ) -> tuple[dict[str, Any], bool]:
        """Shape the arguments from the tool's OWN `input_schema`, so a bot can add a
        tool with any arguments without a change here. `tracks` is the one property the
        service owns (the model emits ids, we resolve them); everything else comes from
        the model's flat `action_args_json`, coerced to the type the schema declares."""
        schema = spec.input_schema or {}
        properties: dict[str, Any] = schema.get("properties") or {}
        required: list[str] = schema.get("required") or []
        supplied = cls._parse_action_args(action_args_json)

        arguments: dict[str, Any] = {}
        if "tracks" in properties or "tracks" in required:
            arguments["tracks"] = [track.model_dump() for track in tracks]

        for name, prop in properties.items():
            if name == "tracks" or name not in supplied:
                continue
            value = _coerce_to_schema(supplied[name], prop if isinstance(prop, dict) else {})
            if value is not None:
                arguments[name] = value

        # A required argument we could not produce means the call would be rejected by
        # the bot anyway — better to report honestly than to send a broken call.
        missing = [
            name
            for name in required
            if name not in arguments or (name == "tracks" and not tracks)
        ]
        return arguments, not missing

    @staticmethod
    def _parse_action_args(action_args_json: str) -> dict[str, Any]:
        """The model emits arguments as a JSON object string (flat, and the same shape
        function-calling itself uses). Anything unparseable is treated as absent — the
        required-argument check below then reports the failure honestly."""
        text = (action_args_json or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            logger.warning("could not parse action_args_json: %r", text[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def forget(self, session: AgentSession) -> None:
        """Clear stored conversation memory for a session."""
        await self.memory.clear(self._session_key(AgentRequest(session=session, message="")))

    @staticmethod
    def _session_key(request: AgentRequest) -> str:
        session = request.session
        base = session.guild_id or session.user_id or "global"
        return f"{_MEMORY_VERSION}:{base}"

    @staticmethod
    def _format_prompt(request: AgentRequest) -> str:
        who = request.session.user_name or "Unknown user"
        prompt = f"{who} says: {request.message}"

        # Render EVERY field the bot sent. This used to be a hardcoded list of
        # known keys, which silently dropped everything else (`volume` never
        # reached the model, so it truthfully answered "I can't see the volume").
        # A new context field must work without a change here.
        parts: list[str] = []
        for key, value in (request.context or {}).items():
            if value is None or value == "" or value == []:
                continue
            if key == "queue_len" and (request.context or {}).get("queue"):
                continue  # the queue itself is listed, its length would be noise
            label = _CONTEXT_LABELS.get(key) or key.replace("_", " ").capitalize()
            rendered = (
                "; ".join(str(item) for item in value)
                if isinstance(value, (list, tuple))
                else str(value)
            )
            parts.append(f"{label}: {rendered}")

        if parts:
            prompt += "\n\nCurrent player state:\n" + "\n".join(parts)
        return prompt
