from typing import Literal

from pydantic import BaseModel, Field

from .track import Track

AgentAction = Literal["play", "enqueue", "replace_queue", "clarify", "none"]


class AgentSession(BaseModel):
    """Identifies who/where the request comes from. Used later for per-guild
    memory; for now it only personalises the prompt."""

    guild_id: str | None = None
    channel_id: str | None = None
    user_id: str | None = None
    user_name: str | None = None


class AgentRequest(BaseModel):
    session: AgentSession = Field(default_factory=AgentSession)
    message: str
    context: dict | None = Field(
        default=None,
        description="Optional current player state from the bot (now_playing, queue_len, ...).",
    )


class AgentResponse(BaseModel):
    spoken_answer: str = Field(
        description="Short, natural, conversational English sentence for TTS. No JSON, ids or lists."
    )
    display_text: str = Field(description="Short text for the Discord chat; may include titles/emoji.")
    action: AgentAction = Field(description="What the bot should do with `tracks`.")
    tracks: list[Track] = Field(
        default_factory=list,
        description="Tracks to act on, taken verbatim from search_tracks results.",
    )
    clarification: str | None = Field(
        default=None,
        description="A follow-up question when action == 'clarify'.",
    )
