"""
HeyGen Avatar with Streaming LLM Example

This example demonstrates how to use HeyGen's avatar streaming
with a regular streaming LLM. This approach has much lower latency
than using Realtime LLMs because text goes directly to HeyGen
without any transcription round-trip.

HeyGen handles all TTS and lip-sync based on the LLM's text output.
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, heygen
from vision_agents.plugins.heygen import VideoQuality

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create agent with HeyGen avatar and streaming LLM."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Assistant with Avatar", id="agent"),
        instructions=(
            "You're a friendly and helpful AI assistant. "
            "Keep your responses conversational and engaging. "
            "Don't use special characters or formatting."
        ),
        # Use regular streaming LLM (not Realtime) for lower latency
        llm=gemini.LLM("gemini-2.0-flash-exp"),
        # Add STT for speech input
        stt=deepgram.STT(),
        # Add HeyGen avatar as a video publisher
        # Note: mute_llm_audio is not needed since streaming LLM doesn't produce audio
        processors=[
            heygen.AvatarPublisher(
                avatar_id="default",  # Use your HeyGen avatar ID
                quality=VideoQuality.HIGH,  # Video quality: VideoQuality.LOW, VideoQuality.MEDIUM, or VideoQuality.HIGH
                resolution=(1920, 1080),  # Output resolution
                mute_llm_audio=False,  # Not needed for streaming LLM
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
        logger.info("Demo opened")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
