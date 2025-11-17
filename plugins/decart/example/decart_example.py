"""
Decart Plugin Example

This is a sample example demonstrating how to structure a plugin example.
You can customize this to showcase your plugin's functionality.
"""

import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import decart, getstream, gemini

logger = logging.getLogger(__name__)

load_dotenv()

processor = decart.RestylingProcessor(initial_prompt="Studio Ghibli animation style", model="mirage_v2")
async def create_agent(**kwargs) -> Agent:
    """Create the agent with your plugin configuration."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI", id="agent"),
        instructions="Be nice to the user",
        llm=gemini.LLM(model="gemini-2.5-flash-lite"),
        # tts=elevenlabs.TTS(),
        # stt=elevenlabs.STT(),
        processors=[processor]
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(8)
        await processor.set_prompt("Arcane animation style")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))

