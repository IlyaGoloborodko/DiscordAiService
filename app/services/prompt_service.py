import os
from typing import Any

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.data.models import PromptInfo, MusicAgentOutput
from .tts_service import text_to_speech

class PromptService:
    def __init__(self):
        self.tm_base_url = os.getenv("TM_BASE_URL")
        self.tm_api_key = os.getenv("TM_API_KEY")

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            base_url=self.tm_base_url,
            api_key=self.tm_api_key
        )

    async def get_text_response(self, prompt_info: PromptInfo) -> AgentRunResult[Any]:
        model = OpenAIChatModel(
            model_name="google/gemma-3-4b",
            provider=self._provider(),
        )
        agent = Agent(model)
        question = f'{prompt_info.user_name or "Unknown user"} asks: {prompt_info.user_message}'
        return await agent.run(question)

    async def get_agent_response_with_search_str(self, prompt_info: PromptInfo) -> AgentRunResult[Any]:
        model = OpenAIChatModel(
            model_name="google/gemma-3-4b",
            provider=self._provider(),
        )
        agent = Agent(
            model,
            output_type=MusicAgentOutput,
            retries=5,
        )
        question = f'{prompt_info.user_name or "Unknown user"} asks: {prompt_info.user_message}'
        return await agent.run(question)

    async def process_prompt(self, prompt_info: PromptInfo):
        agent_response = await self.get_text_response(prompt_info)

        async for chunk in text_to_speech(agent_response.output):
            yield chunk
