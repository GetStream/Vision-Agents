import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
A voice agent whose whole pipeline runs in the Go acceleration backend.

agent.yaml names the agent and instructions.md says what it is, and the router says who
does the work: a config that names no models transcribes, answers and speaks on the
deployment's defaults, and hands anything worth thinking about to the quality tier while
the conversation carries on. The directory is stored on joining, so editing it is the
whole of changing the agent.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""


async def create_agent(**kwargs) -> Agent:
    return Agent(
        config="simple_voice_ai",
        # What this call's spend is filed under, so a bill can be read by environment,
        # customer or feature rather than as one number.
        cost_tracking={"env": "production"},
        # Which memories the agent may recall: whoever it is talking to, and nobody else.
        memory_filter={"user_id": "123"},
    )


async def join_call(agent: Agent, call_id: str, **kwargs) -> None:
    async with agent.join(call_id):
        await agent.responses.create("greet the user in one short sentence")


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
