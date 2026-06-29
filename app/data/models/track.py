from pydantic import BaseModel, Field


class Track(BaseModel):
    """A media item as returned by the search-service. Mirrors its schema so the
    agent can pass tracks through unchanged; the Discord bot resolves the actual
    stream URL just-in-time before playback."""

    provider: str = "youtube"
    id: str = Field(description="Provider-local id, used by the bot to resolve a stream.")
    title: str
    uploader: str | None = None
    url: str | None = None
    duration: float | None = Field(default=None, description="Length in seconds.")
    thumbnail: str | None = None
