from fastapi import APIRouter

from app.data.models import AgentRequest, AgentResponse
from app.services import AgentService

agent_router = APIRouter()


@agent_router.post("/agent")
async def process_agent_message(request: AgentRequest) -> AgentResponse:
    return await AgentService().run(request)
