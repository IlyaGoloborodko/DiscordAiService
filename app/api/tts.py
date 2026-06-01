from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from app.services import text_to_speech

tts_router = APIRouter()


class TtsRequest(BaseModel):
    text: str

@tts_router.post("/tts")
async def process_user_text_prompt(tts_request: TtsRequest):
    return StreamingResponse(
        text_to_speech(tts_request.text),
        media_type="application/octet-stream"
    )