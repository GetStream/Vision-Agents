"""
FunASR STT Example

This example demonstrates how to use the FunASR STT plugin
for self-hosted speech-to-text transcription.
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import elevenlabs, funasr, gemini, getstream, vogent

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with FunASR STT configuration."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="FunASR AI", id="agent"),
        instructions="Be helpful and respond naturally to the user's speech.",
        llm=gemini.LLM("gemini-flash-lite-latest"),
        tts=elevenlabs.TTS(),
        stt=funasr.STT(
            model="iic/SenseVoiceSmall",
            language="auto",
            device="cpu",  # Use "cuda" if you have GPU support
        ),
        turn_detection=vogent.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting FunASR Agent...")

    async with agent.join(call):
        logger.info("Joining call")
        logger.info("FunASR STT ready")

        await asyncio.sleep(5)
        await agent.simple_response(text="Say hi")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
