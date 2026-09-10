import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

dispatch = acceleration.StreamDispatch()


async def create_agent() -> Agent:
    return Agent(config="chat_support")


@dispatch.wait_for_message()
async def inbound_message(message: acceleration.InboundMessage):
    agent = await dispatch.get_or_create_agent(message, create_agent)
    await agent.responses.create(message.text)


async def main() -> None:
    logger.info("waiting for messages; write to an agent channel to start one")
    await dispatch.run()


if __name__ == "__main__":
    asyncio.run(main())
