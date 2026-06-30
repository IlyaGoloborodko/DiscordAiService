import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.data.models import AgentRequest, AgentResponse, Track
from app.services.memory import MemoryStore
from app.services.search_client import SearchClient

SYSTEM_PROMPT = """\
You are a voice DJ assistant for a Discord music bot. The user talks to you (often \
in Russian); you decide what music to play.

Always use the tools to find real tracks before recommending anything — never \
invent track ids. Pick the most fitting tool for the request:
- `search_tracks(query)`: an explicitly named track or artist to play, or plain \
free-text search. Call it as many times as needed (e.g. once per artist or genre).
- `get_playlist_tracks(url)`: the user gives a playlist link, or asks for a \
specific existing playlist — expand it into its tracks instead of searching.
- `get_similar_tracks(artist, track)`: "something new in the style of X", "like \
X", "recommend me something" — pass the artist AND a representative seed track (a \
well-known song by that artist); both are needed. Derive them from the user's \
words or the current player state.
- `get_top_charts(tag?, country?)`: "what's popular / trending / top", or "an hour \
of <genre>/<mood>" — use `tag` for a genre or mood, `country` when the user names \
one; omit both for the global top.

Charts and recommendations usually return several tracks -> prefer action "enqueue".

Choose the action:
- "play": user wants one track / to start playing now -> put the single best track in `tracks`.
- "enqueue": user wants several tracks / a playlist / "an hour of music" -> put multiple tracks in `tracks`.
- "replace_queue": user explicitly wants to replace what's playing with a new set.
- "clarify": the request is too vague to act on -> set `clarification` with a short follow-up question, leave `tracks` empty.
- "none": user isn't asking for music (small talk) -> just answer, leave `tracks` empty.

`tracks` must contain only items returned by the tools, with id/title/url unchanged.
`spoken_answer`: short, natural, conversational ENGLISH sentence for text-to-speech (no JSON, ids or lists).
`display_text`: short text for the Discord chat; titles and emoji are welcome.
"""


@dataclass
class AgentDeps:
    search: SearchClient


class AgentService:
    """Builds and runs the music agent. The model/provider are read from env at
    request time (env is loaded after import), matching the existing services."""

    def _build_agent(self) -> Agent[AgentDeps, AgentResponse]:
        model = OpenAIChatModel(
            model_name=os.getenv("TM_MODEL_NAME"),
            provider=OpenAIProvider(
                base_url=os.getenv("TM_BASE_URL"),
                api_key=os.getenv("TM_API_KEY"),
            ),
        )
        agent = Agent(
            model,
            output_type=AgentResponse,
            deps_type=AgentDeps,
            retries=5,
            system_prompt=SYSTEM_PROMPT,
        )

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
            source needs both. Use for "something like X" / "more in this style"."""
            return await ctx.deps.search.similar(artist, track, limit)

        @agent.tool
        async def get_top_charts(
            ctx: RunContext[AgentDeps],
            tag: str | None = None,
            country: str | None = None,
            limit: int = 10,
        ) -> list[Track]:
            """Popular/trending tracks. `tag` for a genre or mood (e.g. "rock",
            "phonk", "chill"), `country` when the user names one; omit both for the
            global top."""
            return await ctx.deps.search.charts(tag, country, limit)

        return agent

    def __init__(self) -> None:
        self.memory = MemoryStore()

    async def run(self, request: AgentRequest) -> AgentResponse:
        agent = self._build_agent()
        deps = AgentDeps(search=SearchClient())

        session_key = self._session_key(request)
        history = await self.memory.load(session_key)

        result = await agent.run(
            self._format_prompt(request),
            deps=deps,
            message_history=history,
        )

        await self.memory.save(session_key, result.all_messages())
        return result.output

    @staticmethod
    def _session_key(request: AgentRequest) -> str:
        session = request.session
        return session.guild_id or session.user_id or "global"

    @staticmethod
    def _format_prompt(request: AgentRequest) -> str:
        who = request.session.user_name or "Unknown user"
        prompt = f"{who} says: {request.message}"
        if request.context:
            prompt += f"\n\nCurrent player state: {request.context}"
        return prompt
