#!/usr/bin/env python3
"""
Example: Speech-to-Text with NVIDIA Nemotron using Agent class

Requires the Nemotron server running separately (see ../server/).

Usage::
    # First start the server (in separate terminal with NeMo installed):
    cd plugins/nemotron/server
    python nemotron_server.py

    # Then run this example:
    python main.py
"""

import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import nemotron, getstream, gemini, elevenlabs, vogent

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Nemotron Bot", id="agent"),
        instructions="You're a helpful voice assistant. Keep responses short.",
        llm=gemini.LLM(model="gemini-2.0-flash"),
        tts=elevenlabs.TTS(),
        stt=nemotron.STT(
            server_url="http://localhost:8765",
        ),
        turn_detection=vogent.TurnDetection(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    with await agent.join(call):
        await agent.simple_response(
            "Hello! I'm using NVIDIA Nemotron for speech recognition."
        )
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
