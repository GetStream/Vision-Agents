"""
Inworld AI Realtime — tool calling example.

Speech-to-speech agent with function calling. Demonstrates two tools the
model genuinely can't answer on its own, so tool invocations are audible
in the conversation:

- ``get_time`` returns the current ISO-8601 time (no args).
- ``get_weather(city)`` returns a mock forecast so the demo works offline.

Requirements:
- INWORLD_API_KEY environment variable with a plan that permits tool calling
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import datetime
import logging
import random

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, inworld, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


INSTRUCTIONS = """You are a friendly voice assistant.

- Keep every reply to one short sentence.
- When asked for the time, call get_time and read the result back naturally.
- When asked for the weather, call get_weather with the city name.
- Never guess times or weather — always call the tool.
"""


async def create_agent(**kwargs) -> Agent:
    """Create the Inworld Realtime tool-calling agent."""
    realtime = inworld.Realtime(
        instructions=INSTRUCTIONS,
        voice="Dennis",
        force_tool_calling=True,
    )

    @realtime.register_function(
        description="Get the current time as an ISO-8601 string."
    )
    async def get_time() -> str:
        return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
            timespec="seconds"
        )

    @realtime.register_function(description="Get a short weather forecast for a city.")
    async def get_weather(city: str) -> dict:
        conditions = random.choice(["sunny", "cloudy", "rainy", "windy", "snowy"])
        temp_c = random.randint(-5, 32)
        return {"city": city, "conditions": conditions, "temp_c": temp_c}

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Inworld Agent", id="agent"),
        instructions=INSTRUCTIONS,
        llm=realtime,
        turn_detection=smart_turn.TurnDetection(),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Inworld Realtime tool-calling agent...")
    async with agent.join(call):
        logger.info("Agent joined call %s/%s", call_type, call_id)
        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="Greet the user in one short sentence and invite them to ask about the time or weather."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
