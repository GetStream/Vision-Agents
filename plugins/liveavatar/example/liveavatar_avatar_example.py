"""LiveAvatar example.

Adds a real-time HeyGen avatar to an AI agent. LiveAvatar generates
synchronized lip-synced video and audio from the TTS audio stream.

Required environment variables:
    LIVEAVATAR_API_KEY
    LIVEAVATAR_AVATAR_ID
    STREAM_API_KEY
    STREAM_API_SECRET
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import deepgram, gemini, getstream, liveavatar

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
        avatar=liveavatar.Avatar(),
        llm=gemini.LLM("gemini-3.1-flash-lite-preview"),
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
