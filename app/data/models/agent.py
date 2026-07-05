from typing import Any, Literal

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


class ToolSpec(BaseModel):
    """An action the Discord bot exposes for this turn. Declared by the bot, not
    hardcoded here — the service passes these through to the LLM so a new bot
    action works without code changes."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    session: AgentSession = Field(default_factory=AgentSession)
    message: str
    context: dict | None = Field(
        default=None,
        description="Optional current player state from the bot (now_playing, queue, queue_len).",
    )
    tools: list[ToolSpec] | None = Field(
        default=None,
        description="Bot-declared actions for this turn. Present -> respond with tool_calls; "
        "absent -> respond with the legacy action + tracks shape.",
    )


class ToolCall(BaseModel):
    name: str = Field(description="Name of a bot action to invoke; must be one of the request's tools.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the action. For music actions include `tracks`; "
        "for no-input controls leave empty.",
    )


class ToolCallResponse(BaseModel):
    """Response when the request declared `tools`: the bot executes `tool_calls`."""

    spoken_answer: str = Field(description="Short, natural, conversational sentence for TTS. No markdown.")
    display_text: str = Field(description="Short text for the Discord chat; emoji/markdown allowed.")
    clarification: str | None = Field(default=None, description="A follow-up question if one is needed.")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Actions to run; empty when no action is required (bot keeps playing as-is).",
    )


class AgentResponse(BaseModel):
    """Legacy response when the request did NOT declare `tools`."""

    spoken_answer: str = Field(
        description="Short, natural, conversational English sentence for TTS. No JSON, ids or lists."
    )
    display_text: str = Field(description="Short text for the Discord chat; may include titles/emoji.")
    action: AgentAction = Field(description="What the bot should do with `tracks`.")
    tracks: list[Track] = Field(
        default_factory=list,
        description="Tracks to act on, taken verbatim from search results.",
    )
    clarification: str | None = Field(
        default=None,
        description="A follow-up question when action == 'clarify'.",
    )
