from fastapi import APIRouter

from app.data.models import AgentRequest, AgentResponse, ToolCallResponse
from app.services import AgentService

agent_router = APIRouter()


@agent_router.post("/agent", response_model=None)
async def process_agent_message(request: AgentRequest) -> AgentResponse | ToolCallResponse:
    return await AgentService().run(request)
