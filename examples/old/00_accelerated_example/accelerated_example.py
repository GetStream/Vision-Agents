import logging
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.harness import DefaultHarness
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import getstream, stream

logger = logging.getLogger(__name__)

load_dotenv()

"""
The same agent as example 01, with the pipeline running in the Go acceleration backend.

Python creates the call and registers the functions; the backend joins the call, hears the
caller, answers and speaks. Nothing about the agent changes except which llm it is given,
so transcripts, events and function calling all work as they do locally.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""

INSTRUCTIONS = """
You're a voice AI assistant. Keep responses short and conversational.

You are being spoken through ElevenLabs v3, which performs audio tags rather than reading
them out. Write one where it earns its place - [laughs], [whispers], [excited], [sighs] -
and leave the rest of the line plain. A tag every turn is worse than none.

Anything that needs real thought, hand to a skill rather than guessing at it yourself.
""".strip()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My accelerated AI friend", id="agent"),
        instructions=INSTRUCTIONS,
        llm=stream.Accelerated(
            model="gemini/gemini-3.8-flash",
            stt="deepgram/flux-general-en",
            tts="elevenlabs/eleven_v3_conversational",
            subagent="openai/gpt-5.6-sol",
        ),
        harness=DefaultHarness(),
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
