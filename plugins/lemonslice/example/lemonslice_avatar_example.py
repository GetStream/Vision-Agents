"""LemonSlice Avatar example.

Adds a real-time avatar to an AI agent. LemonSlice generates synchronized
lip-synced video from the TTS audio stream.

Required environment variables:
    LEMONSLICE_API_KEY
    LIVEKIT_URL
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET
    STREAM_API_KEY
    STREAM_API_SECRET
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import cartesia, deepgram, gemini, getstream, lemonslice

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Avatar Agent", id="agent"),
        instructions=(
            "You're a friendly AI assistant with a visual avatar. "
            "Keep responses short and conversational."
        ),
        llm=gemini.LLM("gemini-2.5-flash-lite"),
        tts=cartesia.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
        processors=[
            lemonslice.LemonSliceAvatarPublisher(
                agent_id="your-agent-id",
            ),
        ],
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
