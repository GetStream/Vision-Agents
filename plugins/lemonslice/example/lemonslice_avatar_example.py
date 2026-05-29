"""LemonSlice Avatar example.

Adds a real-time avatar to an AI agent. LemonSlice generates synchronized
lip-synced video and audio from the TTS audio stream.

Required environment variables:
    LEMONSLICE_API_KEY
    LEMONSLICE_AGENT_ID  (or LEMONSLICE_AGENT_IMAGE_URL)
    LIVEKIT_URL
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET
    STREAM_API_KEY
    STREAM_API_SECRET
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import deepgram, gemini, getstream, lemonslice

logger = logging.getLogger(__name__)

load_dotenv()


INSTRUCTIONS = (
    "You're a voice AI assistant. Keep responses short and conversational. "
    "Don't use special characters or formatting. Be friendly and helpful."
)


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Avatar Agent", id="agent"),
        instructions=INSTRUCTIONS,
        avatar=lemonslice.Avatar(),
        llm=gemini.LLM("gemini-flash-lite-latest"),
        tts=deepgram.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("tell me something interesting in a short sentence")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
