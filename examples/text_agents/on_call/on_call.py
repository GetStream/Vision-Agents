import asyncio
import logging
import os

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent that reads the company's Sentry and the reader's own Google Calendar.

`agent.yaml` is where the two are set up, and they are set up differently. Sentry is under
`plugins`: the company connects it once, on the dashboard, and every conversation reads the
same account. Google Calendar is under `user_plugins`: each person connects their own, and
the agent asks for it in the conversation the first time it needs it. In a chat that request
is a `plugin_authorization` attachment, a button; here it is the URL, printed.

A calendar belongs to somebody, so the conversation is opened for an end user. Without one
the agent is offered no calendar at all.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it. The
router needs GOOGLE_CALENDAR_MCP_CLIENT_ID and GOOGLE_CALENDAR_MCP_CLIENT_SECRET; Sentry
registers its own client.
"""

USER_ID = os.environ.get("ON_CALL_USER_ID", "on-call-engineer")

QUESTION = """
Which unresolved Sentry issues are newest, and when am I free today to look at them?
"""


async def main() -> None:
    agent = Agent(
        config="on_call", llm=stream.Accelerated(config="on_call", user_id=USER_ID)
    )

    async with agent.chat():
        question = QUESTION.strip()
        while question:
            connect = ""
            async for event in agent.ask(question):
                if event.type == "agent_speech_delta":
                    print(event.text, end="", flush=True)
                elif event.type == "authorization_required":
                    connect = event.url
                    print(f"\n[{event.text}: {event.url}]")
                elif event.type == "error":
                    print(f"[{event.error}]")
            print()

            question = ""
            if connect:
                await asyncio.to_thread(
                    input, "Press Enter once it is connected, to ask again. "
                )
                question = QUESTION.strip()


if __name__ == "__main__":
    asyncio.run(main())
