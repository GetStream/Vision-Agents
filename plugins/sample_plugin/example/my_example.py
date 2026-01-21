"""
Sample Plugin Example

This is a sample example demonstrating how to structure a plugin example.
You can customize this to showcase your plugin's functionality.
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import aws, cartesia, deepgram, getstream, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with your plugin configuration."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Friendly AI", id="agent"),
        instructions="Be nice to the user",
        llm=aws.LLM(model="qwen.qwen3-32b-v1:0"),
        tts=cartesia.TTS(),
        stt=deepgram.STT(),
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

    logger.info("🤖 Starting Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await asyncio.sleep(5)
        await agent.llm.simple_response(text="Say hi")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
