"""
Deepgram TTS Example

This example demonstrates Deepgram TTS integration with Vision Agents.

This example creates an agent that uses:
- Deepgram for text-to-speech (TTS)
- Deepgram for speech-to-text (STT)
- GetStream for edge/real-time communication
- Gemini for LLM

Requirements:
- DEEPGRAM_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- GOOGLE_API_KEY environment variable (for Gemini)
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Deepgram TTS and STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Deepgram Agent", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        tts=deepgram.TTS(),  # Uses Deepgram Aura for text-to-speech
        stt=deepgram.STT(),  # Uses Deepgram Flux for speech-to-text
        llm=gemini.LLM("gemini-2.0-flash"),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Deepgram TTS Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Hello! How can I help you today?")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
