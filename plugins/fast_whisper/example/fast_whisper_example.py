"""
Fast Whisper STT Example

This example demonstrates how to use the Fast Whisper STT plugin
for real-time speech-to-text transcription.
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import elevenlabs, fast_whisper, gemini, getstream, vogent

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Fast Whisper STT configuration."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Fast Whisper AI", id="agent"),
        instructions="Be helpful and respond naturally to the user's speech.",
        llm=gemini.LLM("gemini-2.5-flash-lite"),
        tts=elevenlabs.TTS(),
        stt=fast_whisper.STT(
            model_size="tiny",  # Use base for good balance of speed and accuracy
            device="cpu",  # Use "cuda" if you have GPU support
        ),
        turn_detection=vogent.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Fast Whisper Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("Fast Whisper STT ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Say hi")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
