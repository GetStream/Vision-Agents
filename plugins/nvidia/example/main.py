"""
NVIDIA VLM Example

Creates an agent that uses:
- NVIDIA VLM for vision language model (via Chat Completions API)
- Deepgram for speech-to-text (STT)
- ElevenLabs for text-to-speech (TTS)
- GetStream for edge/real-time communication

Requirements:
- NVIDIA_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
- ELEVENLABS_API_KEY environment variable
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, nvidia

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create the agent with NVIDIA VLM."""
    llm = nvidia.VLM(
        model="nvidia/cosmos-reason2-8b",
        fps=1,
        frame_buffer_seconds=10,
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="NVIDIA Video Assistant", id="agent"),
        instructions="You're a helpful video AI assistant. Analyze the video frames and respond to user questions about what you see. Keep responses concise and descriptive.",
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
        processors=[],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting NVIDIA VLM Agent...")

    async with agent.join(call):
        logger.info("Joining call")
        await agent.edge.open_demo(call)
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
