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
    name: str = Field(description="Name of a bot action to invoke; one of the request's tools.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments shaped by the tool's own `input_schema`, built by the "
        "service — NOT a fixed shape. `tracks` is filled from the model's track ids; "
        "any other property comes from the model's `action_args_json`.",
    )


# --- LLM output (drafts) -------------------------------------------------------
# Deliberately FLAT: a single action string + a flat list of track-id strings.
# Local models (gemma-4-e4b via LM Studio) cannot reliably emit arrays of objects
# in a constrained tool call — nested `tracks:[{id,title}]` fails the gemma parser,
# only a flat `track_ids:[str]` array works. The service maps ids back to full
# tracks (from what the search tools returned), which also drops any invented id.

class ToolCallDraft(BaseModel):
    """What the LLM generates in tool-calling mode."""

    display_text: str = Field(description="The single reply. Shown in chat and, cleaned, spoken by TTS.")
    action: str = Field(default="", description='One action name to run, or "" for no action.')
    track_ids: list[str] = Field(
        default_factory=list,
        description="Ids (from search results) to act on. Only for actions that take tracks.",
    )
    action_args_json: str = Field(
        default="",
        description="The action's NON-track arguments as a flat JSON object string, "
        'e.g. {"level": 6}. Use the argument names from the action\'s listed schema. '
        'Empty ("") when the action takes no arguments or only takes tracks.',
    )
    clarification: str | None = Field(default=None, description="A follow-up question if one is needed.")


class AgentDraft(BaseModel):
    """What the LLM generates in legacy mode (no bot tools declared)."""

    display_text: str = Field(description="The single reply. Shown in chat and, cleaned, spoken by TTS.")
    action: AgentAction = Field(description="What the bot should do with the tracks.")
    track_ids: list[str] = Field(
        default_factory=list,
        description="Ids (from search results) to act on.",
    )
    clarification: str | None = Field(default=None, description="A follow-up question when action == 'clarify'.")


class ToolCallResponse(BaseModel):
    """Response to the bot when the request declared `tools`."""

    spoken_answer: str = Field(description="display_text cleaned of emoji/markdown for TTS. Set by the service.")
    display_text: str = Field(description="The assistant reply for the Discord chat; emoji/markdown allowed.")
    clarification: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Legacy response when the request did NOT declare `tools`."""

    spoken_answer: str = Field(description="display_text cleaned of emoji/markdown for TTS. Set by the service.")
    display_text: str = Field(description="The assistant reply for the Discord chat; may include titles/emoji.")
    action: AgentAction = Field(description="What the bot should do with `tracks`.")
    tracks: list[Track] = Field(default_factory=list)
    clarification: str | None = None
