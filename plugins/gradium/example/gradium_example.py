"""
Gradium Plugin Example

Demonstrates how to use the Gradium STT and TTS plugins with Vision Agents.
"""

import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gradium


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Gradium STT and TTS configuration."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Gradium Agent", id="agent"),
        instructions="Be nice to the user. You are a helpful assistant powered by Gradium.",
        # Gradium STT with VAD-based turn detection
        stt=gradium.STT(
            language="en",
            region="eu",
            vad_threshold=0.7,
        ),
        # Gradium TTS with default voice
        tts=gradium.TTS(
            region="eu",
            speed=0.0,  # Normal speed (range: -4.0 faster to 4.0 slower)
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Gradium Agent with STT and TTS...")

    with await agent.join(call):
        logger.info("Joining call with Gradium STT and TTS")
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
