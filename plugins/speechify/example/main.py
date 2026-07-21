#!/usr/bin/env python3
"""
Example: Text-to-Speech with Speechify using the Agent class

This minimal example shows how to:
1. Create an Agent with Deepgram STT, a Gemini LLM and Speechify TTS
2. Join a Stream video call
3. Greet users and respond to spoken input

Run it, join the call in your browser, and speak to the bot.

Usage::
    uv run main.py run

The script looks for the following env vars (see `.env.example`):
    STREAM_API_KEY / STREAM_API_SECRET
    SPEECHIFY_API_KEY
    DEEPGRAM_API_KEY
    GOOGLE_API_KEY
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Runner
from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core.edge.types import User
from vision_agents.plugins import deepgram, gemini, getstream, speechify

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with Deepgram STT, a Gemini LLM and Speechify TTS."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Speechify Voice Bot", id="agent"),
        instructions=(
            "You're a helpful voice AI assistant. "
            "Keep replies short and conversational."
        ),
        stt=deepgram.STT(),
        llm=gemini.LLM(),
        tts=speechify.TTS(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Speechify TTS voice bot")

    async with agent.join(call):
        logger.info("Joined call")
        await asyncio.sleep(3)
        await agent.simple_response(
            "Hello! I'm listening. What would you like to talk about?"
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
