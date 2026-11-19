import asyncio
import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
An agent that rings a candidate and screens them against the role in this
directory, which `agent.sync()` stores before the call is placed.
"""


async def main() -> None:
    agent = Agent(
        config="recruiter_voice",
    )
    await agent.sync()
    async with agent.outbound_call(
        from_=os.environ["OUTBOUND_FROM"],
        to=os.environ["OUTBOUND_TO"],
        call_id="hello",
    ):
        await agent.responses.create(
            "greet the user and let them know you're a friendly AI agent"
        )


if __name__ == "__main__":
    asyncio.run(main())
