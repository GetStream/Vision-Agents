"""Standalone Gemini 3.8 Live smoke test example — no GetStream SFU required.

Tests connecting to Gemini 3.8 Live models (gemini-3.8-live and
gemini-3.8-live-extended-thinking) directly over WebSockets,
exercising text input, audio output, and async tool calling.

Usage:
    uv run plugins/gemini/example/gemini_live_standalone_example.py
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from vision_agents.core.llm.realtime import (
    RealtimeAgentSpeechEnded,
    RealtimeAgentSpeechStarted,
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeAudioOutputDone,
)
from vision_agents.plugins.gemini import (
    DEFAULT_MODEL,
    LIVE_EXTENDED_THINKING_MODEL,
    Realtime,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_model(model_name: str, test_tool: bool = False) -> None:
    """Run a test conversation turn against a Gemini Live model."""
    logger.info("=== Testing model: %s ===", model_name)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment")
        sys.exit(1)

    rt = Realtime(model=model_name, api_key=api_key)

    if test_tool:
        tool_called = False

        @rt.register_function(
            name="get_weather",
            description="Get current weather for a city.",
        )
        async def get_weather(city: str) -> dict[str, str]:
            nonlocal tool_called
            tool_called = True
            logger.info("Tool 'get_weather' invoked for city: %s", city)
            return {"city": city, "condition": "Sunny", "temperature": "22°C"}

    try:
        logger.info("Connecting to Gemini Live WebSocket...")
        await rt.connect()
        logger.info("Connected successfully.")

        prompt = (
            "What is the weather in Amsterdam? Please use the get_weather tool."
            if test_tool
            else "Say hello and give a one-sentence inspiring quote."
        )

        logger.info("Sending prompt: %s", prompt)
        async for _ in rt.simple_response(prompt):
            pass

        logger.info("Collecting model output...")
        timeout_seconds = 15.0 if model_name == LIVE_EXTENDED_THINKING_MODEL else 10.0
        items = await rt.output.collect(timeout=timeout_seconds)

        audio_chunks = [i for i in items if isinstance(i, RealtimeAudioOutput)]
        transcripts = [i for i in items if isinstance(i, RealtimeAgentTranscript)]
        speech_starts = [i for i in items if isinstance(i, RealtimeAgentSpeechStarted)]
        speech_ends = [i for i in items if isinstance(i, RealtimeAgentSpeechEnded)]
        dones = [i for i in items if isinstance(i, RealtimeAudioOutputDone)]

        logger.info(
            "Received: %d audio chunks, %d transcript deltas, %d start events, %d end events, %d done events",
            len(audio_chunks),
            len(transcripts),
            len(speech_starts),
            len(speech_ends),
            len(dones),
        )

        if transcripts:
            full_text = "".join(t.text for t in transcripts)
            logger.info("Model transcript: %s", full_text)

        if test_tool:
            logger.info("Tool called: %s", tool_called)

    finally:
        await rt.close()
        logger.info("Closed session for %s\n", model_name)


async def main() -> None:
    # 1. Test latency-optimized conversational model (default: gemini-3.8-live)
    await test_model(DEFAULT_MODEL, test_tool=False)

    # 2. Test extended thinking model with background reasoning and async tools
    await test_model(LIVE_EXTENDED_THINKING_MODEL, test_tool=True)


if __name__ == "__main__":
    asyncio.run(main())
