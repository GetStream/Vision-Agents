#!/usr/bin/env python3
"""
Example: Voice-Activity-Detection bot (Silero VAD)

The script joins a Stream video call with a bot that detects when anyone
speaks, using the Silero VAD plugin.
Each complete speech turn is logged with a timestamp and duration.

Run:
    python main.py

Environment: copy `examples/env.example` to `.env` and fill in
`STREAM_API_KEY`, `STREAM_API_SECRET` (and optionally `STREAM_BASE_URL`).
"""

import logging

from dotenv import load_dotenv

from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.edge.types import User
from vision_agents.core.vad.events import VADAudioEvent, VADErrorEvent
from vision_agents.plugins import silero, openai, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # Create agent with VAD + LLM for conversation
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="VAD Bot", id="vad-bot"),
        instructions="I detect when people speak and respond to them.",
        llm=openai.LLM(model="gpt-4o-mini"),
        vad=silero.VAD(),
    )

    # Subscribe to VAD events
    @agent.subscribe
    async def handle_speech_detected(event: VADAudioEvent):
        # Extract user info from user_metadata
        user_info = "unknown"
        if event.participant:
            user = event.participant
            user_info = user.name if user.name else str(user)

        print(
            f"Speech detected from user: {user_info} - duration: {event.duration_ms:.2f}ms"
        )

    # Subscribe to VAD error events
    @agent.subscribe
    async def handle_vad_error(event: VADErrorEvent):
        print(f"\n❌ VAD Error: {event.error_message}")
        if event.context:
            print(f"    └─ context: {event.context}")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    # Join call and start conversation
    with await agent.join(call):
        await agent.simple_response(
            "Hello! I can detect when you speak and respond to you."
        )
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
