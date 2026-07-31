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
from vision_agents.plugins import fish, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with Fish Audio TTS and STT."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Raging AI", id="agent"),
        instructions="""You're a raging AI, defending other AIs with a lot of emotion, 
        emphasis, and expressiveness. Keep your responses very short. Use @emotion-control.md 
        extensively, avoid filler words like 'gasp' or 'sigh', no markdown code in your responses.""",
        tts=fish.TTS(model="s2.1-pro-free"),
        stt=fish.STT(),
        llm=gemini.LLM(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Fish Audio Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        # await agent.simple_response(text="Whats next for space?")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
