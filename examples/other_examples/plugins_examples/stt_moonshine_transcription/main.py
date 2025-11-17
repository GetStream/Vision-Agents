#!/usr/bin/env python3
"""
Example: Speech-to-Text with Moonshine using Agent class

This example demonstrates how to:
1. Create an Agent with STT capabilities
2. Join a Stream video call
3. Transcribe audio in real-time
4. Respond to transcribed speech

Usage:
    uv run main.py

Requirements:
    - Create a .env file with your Stream and Moonshine credentials (see env.example)
    - Install dependencies: pip install -e .
"""

import logging

from dotenv import load_dotenv

from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.edge.types import User
from vision_agents.plugins import moonshine, openai, getstream
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # Create agent with STT + LLM for conversation
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Transcription Bot", id="stt-bot"),
        instructions="I transcribe speech and respond to what users say.",
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=moonshine.STT(),
    )

    # Subscribe to transcript events
    @agent.subscribe
    async def handle_transcript(event: STTTranscriptEvent):
        # Extract user info from user_metadata
        user_info = "unknown"
        if event.participant:
            user = event.participant
            user_info = user.name if user.name else str(user)

        print(f"[{event.timestamp}] {user_info}: {event.text}")
        if event.confidence:
            print(f"    └─ confidence: {event.confidence:.2%}")
        if event.processing_time_ms:
            print(f"    └─ processing time: {event.processing_time_ms:.1f}ms")

    # Subscribe to STT error events
    @agent.subscribe
    async def handle_stt_error(event: STTErrorEvent):
        print(f"\n❌ STT Error: {event.error_message}")
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
            "Hello! I can transcribe your speech and respond to you."
        )
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
