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
A voice agent that speaks with Fish Audio Drama 3 through the Go acceleration backend.

agent.yaml pins the voice to fish/drama-3, which acts bracketed stage directions such as
[laughs] or [warmly] rather than reading them out, so the router tells the model it may
write them. Everything else runs on the router's defaults.

Needs a router with FISH_API_KEY: see acceleration/README.md, then point
STREAM_ACCELERATION_URL at it.
"""


async def create_agent() -> Agent:
    name = "drama_voice"
    return Agent(
        config=name,
        llm=stream.Accelerated(
            config=name, log_latency=os.getenv("VOICE_LATENCY_DAG") == "1"
        ),
        cost_tracking={"env": "development"},
    )


async def main() -> None:
    agent = await create_agent()
    await agent.sync()
    call = await agent.create_call("agent", str(uuid4()))
    await agent.edge.open_demo(call)
    async with agent.join(call):
        await agent.responses.create("greet the user in one short, expressive sentence")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(main())
