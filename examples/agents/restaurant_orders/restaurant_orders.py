import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
An agent that answers the restaurant's phone and takes orders from the menu in
this directory, which is stored on the first call rather than by hand.
"""

dispatch = acceleration.StreamDispatch()


@dispatch.wait_for_call()
async def inbound_call(call: acceleration.CallContext):
    agent = Agent(
        config="restaurant_orders",
    )
    async with agent.join(call):
        await call.wait_for_phone_participant()
        await agent.responses.create(
            "greet the user and let them know you're a friendly AI agent"
        )


async def main() -> None:
    logger.info("waiting for calls; ring the number to start one")
    await dispatch.run()


if __name__ == "__main__":
    asyncio.run(main())
