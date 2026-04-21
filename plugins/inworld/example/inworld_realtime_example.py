"""
Inworld AI Realtime Example

Speech-to-speech agent powered by Inworld's Realtime API (WebRTC transport).

Requirements:
- INWORLD_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import datetime
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, inworld, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the Inworld Realtime agent."""
    realtime = inworld.Realtime(
        instructions="You are a friendly voice assistant. Keep replies concise.",
    )

    @realtime.register_function(
        description="Get the current time as an ISO-8601 string."
    )
    async def get_time() -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Inworld Agent", id="agent"),
        instructions="You are a friendly voice assistant.",
        llm=realtime,
        turn_detection=smart_turn.TurnDetection(),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Inworld Realtime agent...")
    async with agent.join(call):
        logger.info("Agent joined call %s/%s", call_type, call_id)
        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="You are a voice assistant created by InWorld AI. Say hello and introduce yourself in one sentence."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
