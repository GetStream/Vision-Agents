#!/usr/bin/env python3
"""
Example: Text-to-Speech with Cartesia using Agent class

This minimal example shows how to:
1. Create an Agent with TTS capabilities
2. Join a Stream video call
3. Greet users when they join

Run it, join the call in your browser, and hear the bot greet you 🗣️

Usage::
    python main.py

The script looks for the following env vars (see `env.example`):
    STREAM_API_KEY / STREAM_API_SECRET
    CARTESIA_API_KEY
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Runner
from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core.edge.types import User
from vision_agents.plugins import cartesia, deepgram, getstream, openai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # Create agent with TTS
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Narrator", id="agent"),
        instructions="You're the narrator of a story. When you're given a topic start narrating a story and make heavy use of the audio markup tags to customize the speech output that are described in @sonic3-info.md.",
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    # Join call and wait
    async with agent.join(call):
        await asyncio.sleep(3)
        await agent.simple_response("narrate a story about a dragon")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
