import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.plugins import stream as acceleration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

dispatch = acceleration.StreamDispatch()


@dispatch.wait_for_message()
async def inbound_message(message: acceleration.InboundMessage):
    # Only a message nothing is running for arrives here. One written to an agent that is
    # still on a call is answered by the router from that call's own session, which knows
    # what has been said so far and would otherwise answer as though it had not heard it.
    #
    # The session is given the channel as its agent id, which is what puts the answer back
    # in the conversation the question was asked in.
    async with acceleration.TextSession(
        config_id=message.config_id or "chat_support",
        agent_id=message.agent_id,
    ) as session:
        async for _ in session.ask(message.text):
            pass


async def main() -> None:
    await acceleration.sync_agent("chat_support")
    logger.info("waiting for messages; write to an agent channel to start one")
    await dispatch.run()


if __name__ == "__main__":
    asyncio.run(main())
