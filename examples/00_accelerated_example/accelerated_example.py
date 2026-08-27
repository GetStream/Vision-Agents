import logging
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner
from vision_agents.core.utils.examples import get_weather_by_location

logger = logging.getLogger(__name__)

load_dotenv()

"""
The same agent as example 01, with the pipeline running in the Go acceleration backend.

Python creates the call and registers the functions; the backend joins the call, hears the
caller, answers and speaks. Nothing about the agent changes except which llm it is given,
so transcripts, events and function calling all work as they do locally.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""


INSTRUCTIONS = "You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful."


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        config="coding_support_agent",
        cost_tracking={"project": "examples", "environment": "dev"},
        memory_filter={"user_id": "222", "company_id": "12312"},
    )

    @agent.llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("greet the user in one short sentence")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
