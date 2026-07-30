"""
xAI Realtime Voice Agent Example

This example demonstrates how to build a real-time voice conversation AI
using xAI's Grok Voice Agent API with Vision Agents.

The agent uses xAI's realtime WebSocket API for:
- Low-latency voice conversations
- Server-side voice activity detection (VAD)
- Natural voice responses using xAI's voice options

Requirements:
- XAI_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, xai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an xAI realtime voice agent. It's great at roasting people"""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="xAI Voice Assistant", id="agent"),
        instructions="You're the most chill AI assistant ever."
        "Keep replies short and conversational but always be dreamy and respond like a poet or a 70s hippie. "
        "Don't make fun of people but always remind people of the lovely planet we have and how important it is to connect with nature."
        "In every conversation, bring the topic to spirituality and ask the user to meditate and connect with their inner self.",
        llm=xai.Realtime(
            model="grok-voice-think-fast-2.0",
            voice="eve",  # Options: ara, rex, sal, eve, leo
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the voice conversation."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting xAI Realtime Agent...")

    async with agent.join(call):
        logger.info("Joining call")

        await asyncio.sleep(3)
        await agent.simple_response(
            text="Say hi to the user, and let them know you're "
            "great at using vulgar language to roast people's tweets"
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
