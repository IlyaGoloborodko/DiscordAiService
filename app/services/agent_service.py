import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.data.models import AgentRequest, AgentResponse, Track
from app.services.search_client import SearchClient

SYSTEM_PROMPT = """\
You are a voice DJ assistant for a Discord music bot. The user talks to you (often \
in Russian); you decide what music to play.

Always use the `search_tracks` tool to find real tracks before recommending \
anything — never invent track ids. Call it as many times as needed (e.g. once per \
artist or genre, or several times to assemble a playlist).

Choose the action:
- "play": user wants one track / to start playing now -> put the single best track in `tracks`.
- "enqueue": user wants several tracks / a playlist / "an hour of music" -> put multiple tracks in `tracks`.
- "replace_queue": user explicitly wants to replace what's playing with a new set.
- "clarify": the request is too vague to act on -> set `clarification` with a short follow-up question, leave `tracks` empty.
- "none": user isn't asking for music (small talk) -> just answer, leave `tracks` empty.

`tracks` must contain only items returned by `search_tracks`, with id/title/url unchanged.
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

        return agent

    async def run(self, request: AgentRequest) -> AgentResponse:
        agent = self._build_agent()
        deps = AgentDeps(search=SearchClient())
        result = await agent.run(self._format_prompt(request), deps=deps)
        return result.output

    @staticmethod
    def _format_prompt(request: AgentRequest) -> str:
        who = request.session.user_name or "Unknown user"
        prompt = f"{who} says: {request.message}"
        if request.context:
            prompt += f"\n\nCurrent player state: {request.context}"
        return prompt
