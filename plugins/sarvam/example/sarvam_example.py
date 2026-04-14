"""
Sarvam AI Example

This example creates a voice AI agent powered entirely by Sarvam AI models:
- Sarvam STT (saaras:v3) for speech-to-text with VAD-based turn detection
- Sarvam TTS (bulbul:v3) for text-to-speech in Indian languages
- Sarvam LLM (sarvam-m) for chat completions
- GetStream for real-time edge communication

Requirements:
- SARVAM_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, sarvam

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Sarvam STT, TTS, and LLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Sarvam Agent", id="agent"),
        instructions=(
            "You are a helpful multilingual voice assistant. "
            "Reply in the same language the user speaks. "
            "Keep replies short and conversational."
        ),
        stt=sarvam.STT(language="en-IN"),
        tts=sarvam.TTS(language="en-IN", speaker="shubh"),
        llm=sarvam.LLM(model="sarvam-m"),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Sarvam AI Agent...")

    async with agent.join(call):
        logger.info("Joined call")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Hello! How can I help you today?")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
