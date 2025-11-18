#!/usr/bin/env python3
"""
Example: Real-time Call Transcription with Deepgram STT

This example demonstrates how to:
1. Create an Agent with STT capabilities
2. Join a Stream video call
3. Transcribe audio in real-time
4. Respond to transcribed speech using LLMTextResponseCompletedEvent
5. Use AgentSayEvent for TTS synthesis

Usage:
    uv run main.py

Requirements:
    - Create a .env file with your Stream and Deepgram credentials (see env.example)
    - Install dependencies: uv sync
"""

import logging

from dotenv import load_dotenv

from vision_agents.core.agents import Agent, AgentLauncher
from vision_agents.core import cli
from vision_agents.core.edge.types import User
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent
from vision_agents.core.llm.events import LLMTextResponseCompletedEvent
from vision_agents.plugins import deepgram, openai, getstream, elevenlabs

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # Create agent with STT + LLM + TTS for conversation
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Transcription Bot", id="stt-bot"),
        instructions="I'm a helpful transcription bot. I listen to what users say, transcribe their speech, and respond conversationally. Keep responses short and friendly.",
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
    )

    # Subscribe to transcript events
    @agent.subscribe
    async def on_my_transcript(event: STTTranscriptEvent):
        # Extract user info from user_metadata
        user_info = "unknown"
        if event.participant:
            user = event.participant
            user_info = user.name if user.name else str(user)

        agent.logger.info(f"[{event.timestamp}] {user_info}: {event.text}")
        if event.confidence:
            agent.logger.info(f"    └─ confidence: {event.confidence:.2%}")
        if event.processing_time_ms:
            agent.logger.info(
                f"    └─ processing time: {event.processing_time_ms:.1f}ms"
            )

        # Generate a response to the transcribed text
        await agent.simple_response(event.text)

    # Subscribe to LLM response completion events
    @agent.subscribe
    async def handle_llm_response_completed(event: LLMTextResponseCompletedEvent):
        if event.llm_response and event.llm_response.text:
            agent.logger.info(f"LLM Response completed: {event.llm_response.text}")
            # Trigger TTS synthesis using the convenient say method
            await agent.say(event.llm_response.text)

    # Subscribe to STT error events
    @agent.subscribe
    async def handle_stt_error(event: STTErrorEvent):
        agent.logger.error(f"STT Error: {event.error_message}")
        if event.context:
            agent.logger.error(f"    └─ context: {event.context}")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    # Join call and start conversation
    with await agent.join(call):
        await agent.say(
            "Hello! I'm your transcription bot. I'll listen to what you say, transcribe it, and respond to you. Try saying something!"
        )
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
