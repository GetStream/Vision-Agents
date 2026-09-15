"""Gemini 3.8 Live Realtime Agent Example.

Demonstrates real-time speech-to-speech with Google Gemini 3.8 Live models:
- Latency-optimized audio-to-audio conversational model (gemini-3.8-live)
- Extended reasoning model with background thinking (gemini-3.8-live-extended-thinking)
- Asynchronous non-blocking tool calling and video forwarding

Requirements:
- GOOGLE_API_KEY or GEMINI_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import datetime
import logging
import os
import random
from typing import Any

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream
from vision_agents.plugins.gemini import (
    DEFAULT_MODEL,
    LIVE_EXTENDED_THINKING_MODEL,
    Realtime,
)
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs: Any) -> Agent:
    """Create a Gemini 3.8 Realtime voice and video agent.

    Set GEMINI_LIVE_MODEL=gemini-3.8-live-extended-thinking in your environment
    to test the background reasoning model with async function execution.
    """
    model_name = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL)

    llm = Realtime(
        model=model_name,
        fps=2,
    )

    @llm.register_function(
        name="get_current_time",
        description="Get the current UTC time formatted as an ISO-8601 string.",
    )
    async def get_current_time() -> dict[str, str]:
        """Return the current time in UTC."""
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        return {"current_utc_time": now}

    @llm.register_function(
        name="get_weather",
        description="Get the current weather and temperature for a given city.",
    )
    async def get_weather(city: str) -> dict[str, Any]:
        """Return mock weather data for a city."""
        conditions = ["Sunny", "Partly Cloudy", "Rainy", "Clear", "Breezy"]
        temp_c = random.randint(15, 28)
        return {
            "city": city,
            "condition": random.choice(conditions),
            "temperature_celsius": temp_c,
            "humidity": f"{random.randint(40, 80)}%",
        }

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="Gemini 3.8 Assistant",
            id="gemini-live-agent",
        ),
        instructions=(
            "You are a friendly, concise voice assistant powered by Gemini 3.8 Live. "
            "Keep replies short and conversational. When asked about time or weather, "
            "use your available tools."
        ),
        llm=llm,
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs: Any) -> None:
    """Join the call and greet the user."""
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent) -> None:
        if event.participant.user.id != "gemini-live-agent":
            await asyncio.sleep(2)
            await agent.simple_response(
                text="Say hello and let the user know you can answer questions or check the weather."
            )

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    logger.info(
        "Starting Gemini Realtime Agent. Default model: %s. Extended thinking: %s",
        DEFAULT_MODEL,
        LIVE_EXTENDED_THINKING_MODEL,
    )
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
