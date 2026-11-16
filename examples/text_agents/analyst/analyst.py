import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent whose subagent may run the code it writes.

`sandbox: daytona` in `agent.yaml` is the whole of it. Which sandbox an agent is allowed is
a property of the agent rather than of one conversation, so it is declared on disk once and
every session created from this directory gets it without asking.

Only the subagent is offered the sandbox: booting one and running code in it takes seconds,
which the model holding the conversation does not have. That is why the declaration names a
subagent too, and why the arithmetic below arrives as delegated work rather than as an
answer straight away.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
Running the code is Daytona, so the router needs DAYTONA_API_KEY.
"""

QUESTION = """
Monthly revenue for the year ran 41200, 43900, 42100, 47800, 51300, 54900, 53100,
58700, 62400, 61100, 67300, 71900. What was the compound monthly growth rate, and
which months went backwards?
"""


async def main() -> None:
    agent = Agent(config="analyst")

    async with agent.chat():
        async for event in agent.ask(QUESTION.strip()):
            if event.type == "agent_speech_delta":
                print(event.text, end="", flush=True)
            elif event.type == "delegated":
                print(f"\n[handed to {event.skill}]")
            elif event.type == "task_settled":
                print(f"[{event.skill} came back]\n{event.text}")
            elif event.type == "error":
                print(f"[{event.error}]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
