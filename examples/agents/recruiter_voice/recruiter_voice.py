import asyncio
import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
An agent that rings a candidate and screens them against the role in this
directory.
"""


async def main() -> None:
    await acceleration.sync_agent("recruiter_voice")

    agent = Agent(
        config="recruiter_voice",
    )
    async with agent.outbound_call(
        from_=os.environ["OUTBOUND_FROM"],
        to=os.environ["OUTBOUND_TO"],
        call_type="default",
        call_id="hello",
    ):
        await agent.simple_response(
            "greet the user and let them know you're a friendly AI agent"
        )
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(main())
