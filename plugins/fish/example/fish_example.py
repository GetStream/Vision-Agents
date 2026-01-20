"""
Fish Audio TTS and STT Example

This example demonstrates Fish Audio TTS and STT integration with Vision Agents.

This example creates an agent that uses:
- Fish Audio for text-to-speech (TTS)
- Fish Audio for speech-to-text (STT)
- GetStream for edge/real-time communication
- Smart Turn for turn detection

Requirements:
- FISH_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import fish, gemini, getstream, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Fish Audio TTS and STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI", id="agent"),
        instructions="You're a voice AI called Elon Musk. Short replies only no special characters. Read @elon.md to stay in character",
        tts=fish.TTS(),  # Uses Fish Audio for text-to-speech
        stt=fish.STT(),  # Uses Fish Audio for speech-to-text
        llm=gemini.LLM("gemini-2.0-flash"),
        turn_detection=smart_turn.TurnDetection(
            buffer_in_seconds=2.0, confidence_threshold=0.5
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Fish Audio Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Whats next for space?")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
