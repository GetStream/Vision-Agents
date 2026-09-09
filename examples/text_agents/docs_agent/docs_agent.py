import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.plugins import stream as acceleration

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent that answers out of what it has read, and nothing else.

Half of what it knows is the knowledge directory next to this file, which `sync_agent`
stores. The other half is published elsewhere: `add_knowledge_url` reads that page, cuts it
into the same passages a document becomes and keeps them under the same name, so the one
lookup the agent does mid-answer covers both.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it. The
knowledge base is TURBOPUFFER_API_KEY on the router and reading a page is EXA_API_KEY.
"""

AGENT = "docs_agent"
# A directory's knowledge lands in a namespace named after the agent, so a page added under
# that name is found by the same lookup rather than a second one.
QUICKSTART = "https://visionagents.ai/introduction/quickstart"

QUESTION = "What do I need to run my first agent, and what does llm-fast mean?"


async def main() -> None:
    synced = await acceleration.sync_agent(AGENT)
    page = await acceleration.add_knowledge_url(AGENT, QUICKSTART)
    print(f"\n{page.url} is {page.state} as {page.passages} passages\n")

    async with acceleration.TextSession(config_id=synced.config.id) as session:
        async for event in session.ask(QUESTION):
            if event.type == "delta":
                print(event.text, end="", flush=True)
            elif event.type == "looked_up":
                print(f"[looked up {event.query!r}: {event.documents} passages]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
