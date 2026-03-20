"""
Grok TTS — Basic Example

A minimal Vision Agents setup that demonstrates Grok text-to-speech
with Deepgram STT, Gemini LLM, and Stream's real-time edge transport.

Requirements (environment variables):
    XAI_API_KEY          — xAI / Grok API key
    DEEPGRAM_API_KEY     — Deepgram STT key
    GOOGLE_API_KEY       — Google Gemini key
    STREAM_API_KEY       — Stream API key
    STREAM_API_SECRET    — Stream API secret
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, smart_turn
from vision_agents.plugins import grok_tts

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with Grok TTS using the default 'eve' voice."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Grok Voice AI", id="agent"),
        instructions=(
            "You are a friendly and helpful voice assistant powered by Grok. "
            "Keep your responses concise and conversational."
        ),
        tts=grok_tts.TTS(voice="eve"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM(),
        turn_detection=smart_turn.TurnDetection(
            silence_duration_ms=2000,
            speech_probability_threshold=0.5,
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join a call and greet the user."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Grok TTS Agent (basic example)...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text="Hello! I'm your voice assistant running on Grok TTS. How can I help?"
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
