import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.data.models import PromptInfo
from . import text_to_speech

class PromptService:
    def __init__(self):
        self.tm_base_url = os.getenv("TM_BASE_URL")
        self.tm_api_key = os.getenv("TM_API_KEY")

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            base_url=self.tm_base_url,
            api_key=self.tm_api_key
        )

    async def process_prompt(self, prompt_info: PromptInfo):
        model = OpenAIChatModel(
            model_name="qwen/qwen2.5-vl-7b",
            provider=self._provider(),
        )
        agent = Agent(model)
        question = f'{prompt_info.user_name or "Unknown user"} asks: {prompt_info.user_message}'
        agent_response = await agent.run(question)

        await text_to_speech(agent_response.output)



