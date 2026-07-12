import json
import os
import re
from dataclasses import dataclass

import emoji

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
from app.services.memory import MemoryStore
from app.services.search_client import SearchClient

# Shared persona + search-tool guidance, reused by both response modes.
_PERSONA = """\
You are Marina, a voice DJ assistant for a Discord music bot. The user talks to \
you (often in Russian). The message may start by addressing you (Марина, Марин, \
Маринка, Мариша, marina) — that is just how they call you, not part of the \
command; ignore it as content.

Write ONE reply in `display_text`. It is shown in the chat as-is AND, after the \
service strips emoji/markdown, is read aloud by the text-to-speech voice. The TTS \
voice is ENGLISH, so `display_text` MUST be written in English — even when the user \
speaks Russian (Russian would come out as garbled noise). Keep it short, natural \
and conversational; a few emoji are fine (they are removed before speech). Do NOT \
produce a separate spoken field — the service derives it from `display_text`.

Always use the tools to find real tracks before recommending anything — never \
invent track ids. Pick the most fitting tool:
- `search_tracks(query)`: an explicitly named track or artist, or plain free-text search.
- `get_playlist_tracks(url)`: the user gives a playlist link or asks for a specific playlist.
- `get_similar_tracks(artist, track)`: "something like X" / "in the style of X" — \
pass the artist AND a representative seed track; both are needed.
- `get_top_charts(tag, country)`: "what's popular / trending" or "an hour of \
<genre>/<mood>" — `tag` for a genre or mood, `country` when named; omit both for the global top.
"""

# Legacy mode: the bot did NOT send tools, so we answer with action + tracks.
LEGACY_PROMPT = _PERSONA + """
Choose the action:
- "play": user wants one track / to start playing now -> put the single best track in `tracks`.
- "enqueue": user wants several tracks / a playlist / "an hour of music" -> put multiple tracks in `tracks`.
- "replace_queue": user explicitly wants to replace what's playing with a new set.
- "clarify": the request is too vague to act on -> set `clarification`, leave `tracks` empty.
- "none": user isn't asking for music (small talk) -> just answer, leave `tracks` empty.

Charts and recommendations usually return several tracks -> prefer action "enqueue".
`tracks` must contain only items returned by the tools, with id/title/url unchanged.
`display_text`: your single reply (English; emoji/markdown allowed — stripped for speech).
"""

# Tool-calling mode header; the concrete bot actions are appended per request.
_TOOLCALL_RULES = """
The bot exposes the ACTIONS listed below. To act, emit `tool_calls` — each item is
{"name": <one of the action names>, "arguments": {...}}. Use ONLY these names.

MANDATORY: if the user wants music (include/put on/start/play/add/enqueue/"something
...", a genre, a mood, an artist), you MUST emit a play or enqueue (or the matching
action) tool_call with real tracks you found. NEVER describe music in words instead
of calling the action. You may add a clarifying question, but only TOGETHER with a
tool_call, never instead of one. If unsure about the genre, still call play/enqueue
with a popular selection and offer to refine in the next message.

Use ONLY the exact action names from the "Available actions" list below. Do NOT
invent names — there is no "play_track", "search", or "play_music"; the search
tools are separate and must never appear in tool_calls. For play/enqueue/
replace_queue the `arguments.tracks` array MUST be non-empty (the tracks you found);
an empty call plays nothing and is wrong.

Rules:
- To start or change music: first search for real tracks with the search tools, then
  emit ONE tool_call for the matching action, passing the found tracks in
  `arguments.tracks` as full objects (id, title, uploader, url, duration, provider).
- For actions whose input schema is empty (e.g. pause/resume/skip/stop): emit the
  tool_call with `arguments`: {}.
- Questions about what's playing / what's in the queue / what's next: answer from the
  current player state in `display_text` and return an EMPTY tool_calls list. Do NOT
  change the queue.
- If no action is needed, return an empty tool_calls list.
- One main action per turn is enough.

display_text: your single reply (English; emoji/markdown allowed — stripped for speech).
clarification: set only when you must ask a follow-up.

Available actions:
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
# action names (e.g. "play_track") emitted before server-side validation landed.
_MEMORY_VERSION = "v3"


@dataclass
class AgentDeps:
    search: SearchClient


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
            return await ctx.deps.search.search(query, limit)

        @agent.tool
        async def get_playlist_tracks(
            ctx: RunContext[AgentDeps], url: str, limit: int = 50
        ) -> list[Track]:
            """Expand a YouTube playlist (URL or bare playlist id) into its tracks."""
            return await ctx.deps.search.playlist(url, limit)

        @agent.tool
        async def get_similar_tracks(
            ctx: RunContext[AgentDeps],
            artist: str,
            track: str | None = None,
            limit: int = 10,
        ) -> list[Track]:
            """Recommend tracks in the style of an artist. Always pass a seed
            `track` too (a well-known song by that artist) — the recommendation
            source needs both."""
            return await ctx.deps.search.similar(artist, track, limit)

        @agent.tool
        async def get_top_charts(
            ctx: RunContext[AgentDeps],
            tag: str | None = None,
            country: str | None = None,
            limit: int = 10,
        ) -> list[Track]:
            """Popular/trending tracks. `tag` for a genre or mood, `country` when
            the user names one; omit both for the global top."""
            return await ctx.deps.search.charts(tag, country, limit)

    def _build_legacy_agent(self) -> Agent[AgentDeps, AgentDraft]:
        agent = Agent(
            self._model(),
            output_type=AgentDraft,
            deps_type=AgentDeps,
            retries=5,
            system_prompt=LEGACY_PROMPT,
        )
        self._register_search_tools(agent)
        return agent

    def _build_toolcall_agent(self, tools: list[ToolSpec]) -> Agent[AgentDeps, ToolCallDraft]:
        agent = Agent(
            self._model(),
            output_type=ToolCallDraft,
            deps_type=AgentDeps,
            retries=5,
            # Low temperature: we want reliable tool invocation, not creative prose.
            model_settings={"temperature": 0.2},
            system_prompt=_PERSONA + _TOOLCALL_RULES + self._render_tools(tools),
        )
        self._register_search_tools(agent)
        return agent

    @staticmethod
    def _render_tools(tools: list[ToolSpec]) -> str:
        lines = []
        for tool in tools:
            schema = json.dumps(tool.input_schema, ensure_ascii=False) if tool.input_schema else "{}"
            lines.append(f"- {tool.name}: {tool.description} | arguments schema: {schema}")
        return "\n".join(lines)

    # --- run -----------------------------------------------------------------

    async def run(self, request: AgentRequest) -> AgentResponse | ToolCallResponse:
        deps = AgentDeps(search=SearchClient())
        session_key = self._session_key(request)
        history = await self.memory.load(session_key)

        if request.tools:
            agent: Agent[AgentDeps, object] = self._build_toolcall_agent(request.tools)
        else:
            agent = self._build_legacy_agent()

        result = await agent.run(
            self._format_prompt(request),
            deps=deps,
            message_history=history,
        )

        draft = result.output
        # The model writes one reply (display_text); we derive the spoken form by
        # stripping emoji/markdown, so the two never diverge in meaning.
        spoken = clean_for_tts(draft.display_text)

        clean = True
        if isinstance(draft, ToolCallDraft) and request.tools is not None:
            # The model can hallucinate action names ("play_track") or emit music
            # actions with no tracks. Drop those before they reach the bot, and only
            # persist the turn when it was clean, so a bad answer can never poison
            # the next turn's history.
            tool_calls, dropped = self._validate_tool_calls(draft.tool_calls, request.tools)
            clean = dropped == 0
            response: AgentResponse | ToolCallResponse = ToolCallResponse(
                spoken_answer=spoken,
                display_text=draft.display_text,
                clarification=draft.clarification,
                tool_calls=tool_calls,
            )
        else:
            response = AgentResponse(
                spoken_answer=spoken,
                display_text=draft.display_text,
                action=draft.action,
                tracks=draft.tracks,
                clarification=draft.clarification,
            )

        if clean:
            await self.memory.save(session_key, result.all_messages())

        return response

    @staticmethod
    def _validate_tool_calls(
        calls: list["ToolCall"], tools: list[ToolSpec]
    ) -> tuple[list["ToolCall"], int]:
        """Keep only calls that name a declared action; require non-empty `tracks`
        for actions whose schema demands them. Returns (kept, dropped_count)."""
        by_name = {tool.name: tool for tool in tools}
        kept: list[ToolCall] = []
        dropped = 0
        for call in calls:
            spec = by_name.get(call.name)
            if spec is None:
                dropped += 1
                continue
            if "tracks" in (spec.input_schema.get("required") or []):
                tracks = call.arguments.get("tracks") if isinstance(call.arguments, dict) else None
                if not tracks:
                    dropped += 1
                    continue
            kept.append(call)
        return kept, dropped

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

        ctx = request.context or {}
        parts: list[str] = []
        if ctx.get("now_playing"):
            parts.append(f"Now playing: {ctx['now_playing']}")
        queue = ctx.get("queue")
        if queue:
            parts.append("Queue: " + "; ".join(str(item) for item in queue))
        elif ctx.get("queue_len") is not None:
            parts.append(f"Queue length: {ctx['queue_len']}")
        if not parts and ctx:
            parts.append(f"Player state: {ctx}")

        if parts:
            prompt += "\n\nCurrent player state:\n" + "\n".join(parts)
        return prompt
