"""
Mistral Voxtral STT Example

This example demonstrates Mistral Voxtral STT integration with Vision Agents.

This example creates an agent that uses:
- Mistral Voxtral for speech-to-text (STT)
- Deepgram for text-to-speech (TTS)
- GetStream for edge/real-time communication
- Smart Turn for turn detection (Mistral STT doesn't have built-in turn detection)
- Gemini for LLM

Requirements:
- MISTRAL_API_KEY environment variable
- DEEPGRAM_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- GOOGLE_API_KEY environment variable (for Gemini)
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, mistral

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Mistral STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Mistral Agent", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        tts=deepgram.TTS(),  # Uses Deepgram for text-to-speech
        stt=mistral.STT(),  # Uses Mistral Voxtral for speech-to-text
        llm=gemini.LLM(),
        # turn_detection=smart_turn.TurnDetection(),  # Required since Mistral STT has no turn detection
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Mistral STT Agent...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(5)

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
