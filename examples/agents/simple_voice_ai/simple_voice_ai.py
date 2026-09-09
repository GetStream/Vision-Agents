import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
A voice agent whose whole pipeline runs in the Go acceleration backend.

instructions.md says what the agent is, and the router says who does the work: a config
that names no models transcribes, answers and speaks on the deployment's defaults, and
hands anything worth thinking about to the quality tier while the conversation carries on.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""


async def create_agent(**kwargs) -> Agent:
    await acceleration.sync_agent("simple_voice_ai")

    return Agent(config="simple_voice_ai")


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    async with agent.join(call_type, call_id):
        await agent.simple_response("greet the user in one short sentence")


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
