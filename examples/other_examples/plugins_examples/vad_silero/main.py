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

import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.core.vad.events import VADAudioEvent, VADErrorEvent
from vision_agents.plugins import silero, openai, getstream

load_dotenv()

async def main():
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
        
        print(f"Speech detected from user: {user_info} - duration: {event.duration_ms:.2f}ms")

    # Subscribe to VAD error events
    @agent.subscribe
    async def handle_vad_error(event: VADErrorEvent):
        print(f"\n❌ VAD Error: {event.error_message}")
        if event.context:
            print(f"    └─ context: {event.context}")

    # Create call and open demo
    call = agent.edge.client.video.call("default", str(uuid4()))
    agent.edge.open_demo(call)

    # Join call and start conversation
    with await agent.join(call):
        await agent.simple_response("Hello! I can detect when you speak and respond to you.")
        await agent.finish()

if __name__ == "__main__":
    asyncio.run(main())
