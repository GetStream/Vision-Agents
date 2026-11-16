import logging
import os
import sys

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner
from vision_agents.plugins import stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

load_dotenv()

"""
A voice agent that speaks with Fish Audio Drama 3 through the Go acceleration backend.

agent.yaml pins the voice to fish/drama-3, which acts bracketed stage directions such as
[laughs] or [warmly] rather than reading them out, so the router tells the model it may
write them. Everything else runs on the router's defaults.

Needs a router with FISH_API_KEY: see acceleration/README.md, then point
STREAM_ACCELERATION_URL at it.
"""


async def create_agent(**kwargs) -> Agent:
    name = "drama_voice"
    return Agent(
        config=name,
        llm=stream.Accelerated(
            config=name, log_latency=os.getenv("VOICE_LATENCY_DAG") == "1"
        ),
        cost_tracking={"env": "development"},
    )


async def join_call(agent: Agent, call_id: str, **kwargs) -> None:
    async with agent.join(call_id):
        await agent.responses.create("greet the user in one short, expressive sentence")


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
