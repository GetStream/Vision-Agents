"""
Transformers Tool Calling Example

Demonstrates tool calling with a local HuggingFace model in an Agent.
The model runs on your hardware (MPS/CUDA/CPU) and can invoke registered
functions, then use the results in its follow-up response.

Creates an agent that uses:
- TransformersLLM for local inference with tool calling
- Deepgram for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable

First run will download the model (~3 GB for Qwen2.5-1.5B-Instruct).
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream
from vision_agents.plugins.huggingface.transformers_llm import TransformersLLM

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with a local Transformers LLM and tool calling."""
    llm = TransformersLLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=200,
    )

    @llm.register_function("get_weather", description="Get current weather for a city")
    async def get_weather(city: str) -> str:
        logger.info(f"  [tool] get_weather called with city={city}")
        return f"Sunny, 22C in {city}"

    @llm.register_function("get_time", description="Get current time in a timezone")
    async def get_time(timezone: str) -> str:
        logger.info(f"  [tool] get_time called with timezone={timezone}")
        return f"14:30 in {timezone}"

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Tool Calling Agent", id="agent"),
        instructions=(
            "You are a helpful voice assistant with access to tools. "
            "Use get_weather for weather questions and get_time for time questions. "
            "Always use your tools instead of guessing. "
            "Keep responses short and conversational."
        ),
        llm=llm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
        streaming_tts=True,
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Tool Calling Agent...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="Greet the user and let them know you can check the weather and time."
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
