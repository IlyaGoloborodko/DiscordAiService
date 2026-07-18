from fastapi import APIRouter

from app.data.models import PlaybackReport
from app.recommendations import history

playback_router = APIRouter()


@playback_router.post("/playback")
async def report_playback(report: PlaybackReport) -> dict:
    """The Discord bot reports a track that finished, was skipped or was stopped.

    Called once per track that actually reached the speakers — never for tracks
    that only sat in the queue. That distinction is the whole point: it separates
    "we suggested this" from "they listened to this".

    The bot sends these fire-and-forget, so this must be quick and must never
    make it retry. Storage problems are logged on our side, not pushed back.
    """
    await history.confirm_play(
        guild_id=report.session.guild_id or "",
        track_id=report.track_id,
        played_ms=report.played_ms,
        duration_ms=report.duration_ms,
        reason=report.reason,
        provider=report.provider,
    )
    return {"status": "ok"}
