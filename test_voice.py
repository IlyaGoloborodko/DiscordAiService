from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from wyoming_tests import text_to_speech


async def talk_to_agent(question: str):
    provider = OpenAIProvider(
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed"
    )
    model = OpenAIChatModel(
        model_name="qwen/qwen2.5-vl-7b",
        provider=provider
    )
    agent = Agent(model)
    agent_response = await agent.run(question)

    await text_to_speech(agent_response.output)

