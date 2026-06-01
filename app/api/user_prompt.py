from fastapi import APIRouter

from app.data.models import PromptInfo, MusicAgentOutput
from app.services import PromptService

prompt_router = APIRouter()


@prompt_router.post("/prompt")
async def process_user_text_prompt(prompt: PromptInfo) -> MusicAgentOutput:
    agent_response = await PromptService().get_agent_response_with_search_str(prompt)
    return agent_response.output
