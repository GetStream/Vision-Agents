"""
Transformers Local LLM Example

Demonstrates running a local HuggingFace model with Vision Agents.
The model runs directly on your hardware (MPS/CUDA/CPU) instead of via API.

Creates an agent that uses:
- TransformersLLM for local inference
- Deepgram for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable

First run will download the model (~3GB for Qwen2.5-1.5B).
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, huggingface

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with a local Transformers LLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Local LLM Agent", id="agent"),
        instructions=(
            "You are a voice assistant. Respond in one sentence. "
            "Never use lists, markdown or special formatting."
        ),
        llm=huggingface.TransformersLLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            max_new_tokens=100,
        ),
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
        streaming_tts=True,
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Local Transformers Agent...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(2)
        await agent.llm.simple_response(text="Greet the user warmly.")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
