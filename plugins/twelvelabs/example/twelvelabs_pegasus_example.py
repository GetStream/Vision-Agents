"""TwelveLabs Pegasus video understanding demo.

Runs a Vision Agent that joins a Stream call, buffers a few seconds of the
caller's video, and uses Pegasus to answer questions about what just happened.

Usage:
    uv run plugins/twelvelabs/example/twelvelabs_pegasus_example.py

Then open the join URL printed in the logs in your browser and turn on your
camera; the agent will describe what it sees.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, twelvelabs
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = twelvelabs.PegasusVLM(
        api_key=os.getenv("TWELVELABS_API_KEY"),
    )

    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(name="Pegasus Vision Agent", id="agent"),
        instructions="Describe what just happened in the video.",
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(5)  # let a few seconds of video buffer
            await agent.simple_response("Describe what just happened in the video")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
