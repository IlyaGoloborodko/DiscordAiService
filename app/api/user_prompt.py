from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.data.models import PromptInfo
from app.services import PromptService

prompt_router = APIRouter()


@prompt_router.post("/prompt")
async def user_prompt(prompt: PromptInfo):
    prompt_service = PromptService()
    return StreamingResponse(
        prompt_service.process_prompt(prompt),
        media_type="application/octet-stream"
    )

