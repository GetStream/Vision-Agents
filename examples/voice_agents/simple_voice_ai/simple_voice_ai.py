import asyncio
import logging
import os
import sys
from uuid import uuid4

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

load_dotenv()

"""
A voice agent whose whole pipeline runs in the Go acceleration backend.

agent.yaml names the agent and instructions.md says what it is, and the router says who
does the work: a config that names no models transcribes, answers and speaks on the
deployment's defaults, and hands anything worth thinking about to the quality tier while
the conversation carries on. `agent.sync()` stores the directory before the call, so
editing it and running again is the whole of changing the agent.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
"""


async def create_agent() -> Agent:
    name = "simple_voice_ai"
    return Agent(
        config=name,
        llm=stream.Accelerated(
            config=name, log_latency=os.getenv("VOICE_LATENCY_DAG") == "1"
        ),
        # What this call's spend is filed under, so a bill can be read by environment,
        # customer or feature rather than as one number.
        cost_tracking={"env": "production"},
    )


async def main() -> None:
    agent = await create_agent()
    await agent.sync()
    call = await agent.create_call("agent", str(uuid4()))
    await agent.edge.open_demo(call)
    async with agent.join(call):
        await agent.responses.create("greet the user in one short sentence")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(main())
