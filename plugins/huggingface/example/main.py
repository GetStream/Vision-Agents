"""
HuggingFace LLM Example

This example demonstrates HuggingFace Inference integration with Vision Agents.

Creates an agent that uses:
- HuggingFace for LLM (via Inference Providers API)
- Deepgram for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- HF_TOKEN environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
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
    """Create the agent with HuggingFace LLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="HuggingFace Agent", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        llm=huggingface.LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct", provider="auto"
        ),
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting HuggingFace Agent...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="I am experimenting with running you, an LLM on HuggingFace. Tell me a short story"
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
