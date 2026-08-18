"""ByteDance STT + TTS Example.

Runs a voice agent that uses:
- ByteDance Seed Speech for speech-to-text (STT)
- ByteDance Seed Speech for text-to-speech (TTS)
- GetStream for edge/real-time communication
- Gemini for the LLM

Requirements:
- BYTEDANCE_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- GOOGLE_API_KEY environment variable (for Gemini)
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import bytedance, gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="ByteDance Agent", id="agent"),
        instructions="You're a helpful voice AI assistant. Keep replies short and conversational.",
        stt=bytedance.STT(),
        llm=gemini.LLM(),
        tts=bytedance.TTS(speaker="zh_female_vv_uranus_bigtts"),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.edge.open_demo(call)
        await asyncio.sleep(5)
        await agent.simple_response(text="Hello! How can I help you today?")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
