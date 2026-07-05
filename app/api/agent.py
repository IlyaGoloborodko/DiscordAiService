from fastapi import APIRouter

from app.data.models import AgentRequest, AgentResponse, AgentSession, ToolCallResponse
from app.services import AgentService

agent_router = APIRouter()


@agent_router.post("/agent", response_model=None)
async def process_agent_message(request: AgentRequest) -> AgentResponse | ToolCallResponse:
    return await AgentService().run(request)


@agent_router.post("/agent/forget")
async def forget_session(session: AgentSession) -> dict:
    """Clear stored conversation memory for a session (e.g. a poisoned channel)."""
    await AgentService().forget(session)
    return {"status": "ok"}
