from pydantic import BaseModel, Field

from .agent import AgentSession


class PlaybackReport(BaseModel):
    """The Discord bot telling us a track actually played, and for how long.

    We write a history row the moment we hand tracks to the bot, but that only
    means "queued" — a five-track queue where the listener heard two would count
    all five. This report is what turns "we offered it" into "they heard it".
    Only heard tracks should shape anyone's taste.
    """

    session: AgentSession = Field(default_factory=AgentSession)
    track_id: str = Field(description="Provider-local id, the same one we sent in tool_calls.")
    played_ms: int = Field(description="Real audio time, excluding anything paused.")
    provider: str | None = None
    duration_ms: int | None = Field(default=None, description="Full length of the track.")
    reason: str | None = Field(
        default=None,
        description='Why it stopped: finished / skipped / stopped / disconnected. '
        "Stored as-is — a skip is NOT read as dislike (people skip songs they love "
        "because they just heard them).",
    )
