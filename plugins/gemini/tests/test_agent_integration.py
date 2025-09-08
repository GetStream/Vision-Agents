import asyncio

from dotenv import load_dotenv
import pytest

import os
from typing import List, TypedDict
from stream_agents.plugins import gemini
from stream_agents.core.agents import Agent

load_dotenv()


class _Events(TypedDict):
    audio: List[bytes]
    text: List[str]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_agent_integration_hits_live_endpoints():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key is None:
        pytest.skip("GOOGLE_API_KEY not set – skipping agent+Gemini live integration test")

    # Build Agent with live Gemini Realtime (no RTC join needed to hit Gemini endpoints)
    agent = Agent(
        llm=gemini.Realtime(api_key=api_key),
        instructions="Keep responses short.",
        processors=[],
    )

    # Capture streaming events
    events: _Events = {"audio": [], "text": []}

    @agent.llm.on("audio")  # type: ignore[arg-type]
    async def _on_audio(data: bytes):
        events["audio"].append(data)

    @agent.llm.on("text")  # type: ignore[arg-type]
    async def _on_text(text: str):
        events["text"].append(text)

    # Wait for live session
    ready = await agent.llm.wait_until_ready(timeout=10.0)
    assert ready is True

    # Standardized simple response path
    await agent.llm.simple_response(text="Say a short greeting in English.")

    # Native passthrough path
    await agent.llm.native_send_realtime_input(text="Confirm you can hear me.")

    # Allow time for any responses to arrive
    for _ in range(40):
        if events["audio"] or events["text"]:
            break
        await asyncio.sleep(0.25)

    # We expect at least one audio or text chunk from the live model
    assert events["audio"] or events["text"], "No response received from Gemini Live via Agent"
