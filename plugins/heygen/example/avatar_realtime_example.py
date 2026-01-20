"""
HeyGen Avatar with Realtime LLM Example

This example demonstrates using a HeyGen avatar with a Realtime LLM.
HeyGen provides the lip-synced avatar video based on text transcriptions,
while Gemini Realtime provides the audio directly.
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream, heygen
from vision_agents.plugins.heygen import VideoQuality

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create agent with Gemini Realtime and HeyGen avatar."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Avatar AI Assistant", id="agent"),
        instructions=(
            "You are a helpful AI assistant with a virtual avatar. "
            "Keep responses conversational and natural. "
            "Be friendly and engaging."
        ),
        llm=gemini.Realtime(model="gemini-2.5-flash-native-audio-preview-12-2025"),
        processors=[
            heygen.AvatarPublisher(
                avatar_id="default",
                quality=VideoQuality.HIGH,
            )
        ],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the avatar agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting HeyGen Avatar Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        # Start the conversation
        await agent.llm.simple_response(
            text="Hello! I'm your AI assistant. How can I help you today?"
        )
        logger.info("Greeted the user")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
