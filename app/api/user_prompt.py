from fastapi import APIRouter, BackgroundTasks

from app.data.models import PromptInfo
from app.services import PromptService

prompt_router = APIRouter()


@prompt_router.post("/prompt")
def user_prompt(prompt: PromptInfo):

    prompt_service = PromptService()
    result = prompt_service()

    return {"message": "Hello, World!"}

