import asyncio
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, User
from vision_agents.core.telephony import InboundCall
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import getstream, stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
An agent that answers a real phone call.

The direction is what makes this different from example 13. Nobody here places a call: the
caller dials one of your numbers, the vendor sends it to Stream over SIP, and Stream puts
them in a call and tells the router. The router hands the call down to whichever worker is
waiting, which is this process.

So this connects out and waits, and nothing about this machine has to be reachable from the
internet. The router does, because Stream has to be able to tell it a call arrived.

Needs, once:

    cd acceleration
    go run ./cmd/phone buy -vendor telnyx -number +15125551234
    go run ./cmd/phone attach -number +15125551234 -customer examples
    go run ./cmd/phone hooks -url https://your-tunnel.ngrok.app

Then run this and ring the number.
"""


INSTRUCTIONS = (
    "You are a voice AI assistant answering the phone. Greet the caller, say who you are, "
    "and ask how you can help. Keep replies short and conversational, and do not use "
    "special characters."
)


dispatch = stream.StreamDispatch()


@dispatch.wait_for_call()
async def answer(call: InboundCall) -> None:
    """Hold the conversation for one arriving call.

    Runs once per call, in its own task, so a second caller is answered while the first is
    still talking.
    """
    logger.info("a call from %s on %s", call.caller_number, call.called_number)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="John", id="agent"),
        llm=stream.Accelerated(config="john"),
        cost_tracking={"project": "examples", "environment": "dev"},
    )

    @agent.llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    # The caller is already in the call, so this joins theirs rather than making one. It
    # waits for them to finish joining before returning, so the greeting is not said to an
    # empty call.
    async with agent.answer(call):
        await agent.simple_response("greet the caller and ask how you can help")
        await agent.finish()

    logger.info("the call from %s ended", call.caller_number)


async def main() -> None:
    # The pipeline is decided once and stored under a name, so every call answered by this
    # worker is answered the same way without describing it again per call.
    await stream.define_agent(
        "john",
        instructions=INSTRUCTIONS,
        llm="llm-fast",
        stt="en-low-latency",
        tts="en-low-latency",
    )

    number = os.environ.get("INBOUND_NUMBER")
    if number:
        logger.info("waiting for calls to %s", number)
    logger.info("waiting for calls; ring the number to start one")
    await dispatch.run()


if __name__ == "__main__":
    asyncio.run(main())
