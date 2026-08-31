import asyncio
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from vision_agents.core import Agent, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import getstream, stream

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
An agent that rings a real phone and holds the conversation when it is answered.

Stream's SIP is inbound only, so this is the vendor calling the person and bridging the
answered leg into a Stream call. The agent is in that call before the phone rings, which is
why nobody ever picks up to silence.

The pipeline runs in the Go backend, configured once by name with define_agent and named
here rather than described again.

Needs a router with a telephony vendor configured and a number bought from it: see
acceleration/README.md. Set OUTBOUND_TO to the handset to ring.
"""


INSTRUCTIONS = (
    "You are a voice AI assistant calling somebody. Say who you are in your first "
    "sentence. Keep replies short and conversational, and do not use special characters."
)


async def main() -> None:
    to = os.environ["OUTBOUND_TO"]

    # The whole pipeline is decided once and stored under a name. Everything a session
    # needs is in it, so an agent that uses it names it and says nothing else about models.
    await stream.define_agent(
        "john",
        instructions=INSTRUCTIONS,
        llm="llm-fast",
        stt="en-low-latency",
        tts="en-low-latency",
    )

    phone = stream.Phone()
    held = await phone.numbers()
    if not held:
        raise SystemExit(
            "no numbers to call from; buy one first, e.g. "
            "go run ./cmd/phone buy -vendor twilio -number +15125551234"
        )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="John", id="agent"),
        llm=stream.Accelerated(config="john"),
        phone=phone,
        cost_tracking={"project": "examples", "environment": "dev"},
    )

    @agent.llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    async with agent.outbound_call(
        from_=held[0].e164,
        to=to,
        call_type="default",
        call_id="hello",
        # Long enough for somebody to reach the phone, short enough not to sit in their
        # voicemail talking to it.
        ring_timeout=25.0,
    ) as placed:
        logger.info("ringing %s, join %s to listen in", to, placed.call_id)
        await agent.simple_response(
            "greet the user and let them know you're a friendly AI agent"
        )
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(main())
