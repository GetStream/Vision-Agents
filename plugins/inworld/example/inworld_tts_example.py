"""
Inworld AI TTS Example

This example demonstrates Inworld AI TTS integration with Vision Agents.

This example creates an agent that uses:
- Inworld AI for text-to-speech (TTS)
- Stream for edge/real-time communication
- Deepgram for speech-to-text (STT)
- Smart Turn for turn detection

Requirements:
- INWORLD_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, inworld, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Inworld AI TTS."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI", id="agent"),
        instructions="Read @inworld-audio-guide.md",
        tts=inworld.TTS(voice_id="Ashley"),
        stt=deepgram.STT(),
        llm=gemini.LLM("gemini-2.0-flash"),
        turn_detection=smart_turn.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Inworld AI Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Tell me a story about a dragon.")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
