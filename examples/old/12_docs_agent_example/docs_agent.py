import asyncio
import logging
import sys

from dotenv import load_dotenv
from vision_agents.core.harness import Skill
from vision_agents.plugins import stream

logger = logging.getLogger(__name__)

load_dotenv()

"""
An agent that answers questions about the documentation, in writing.

There is no call here and no audio: the backend runs the same agent a phone call would
get, minus the voice. What it keeps is the part worth showing, which is how it arrives at
an answer. It looks the question up in a knowledge base, and hands anything that deserves
more than a fast model to a slower one through a skill.

Both are configured once, in Postgres, as an agent config: the skills it may delegate to
and the knowledge base it may read. This script writes that config, then talks to it.

Needs a router and a knowledge base filled from the docs:

    cd acceleration && go run ./cmd/knowledge -namespace docs ../docs ../README.md
    uv run docs_agent.py
"""

KNOWLEDGE = "docs"

INSTRUCTIONS = """
You answer questions about the Vision Agents documentation.

Look things up before answering. The documentation is the only thing you know, so a
question you can find no passage for is one to say you cannot answer rather than one to
invent an answer to. Say which file you read it in.

Keep an answer to a few sentences unless the reader asks for more.
""".strip()

# What the fast model may hand to the slower one. The names matter: a skill defined here
# replaces the built-in of the same name, which is how "explain" becomes an explanation
# written to be read rather than one written to be heard on a phone.
SKILLS = [
    Skill(
        name="explain",
        description="a part of the documentation the reader has asked to understand",
        deadline_seconds=25,
        instructions="""
You are the explaining half of a documentation agent. The passages the agent looked up
are in the conversation above; work from those rather than from what you remember about
the library.

Explain the thing asked about and how it fits with the rest. A short code block is worth
more than a paragraph describing one. Name the file each claim came from.

Six sentences at most, plus code. If the passages do not cover it, say so plainly and say
what they do cover.
""".strip(),
    ),
    Skill(
        name="compare",
        description="two ways of doing something, weighed against each other",
        deadline_seconds=30,
        instructions="""
You are the comparing half of a documentation agent. The passages the agent looked up are
in the conversation above.

Say what each option is for and what it costs, then say which one the reader should take
and what would change that. Lead with the recommendation.

If the passages cover only one of the two, say which one is missing rather than guessing
at it.
""".strip(),
    ),
    Skill(
        name="troubleshoot",
        description="an error or a symptom the reader is stuck on",
        deadline_seconds=30,
        instructions="""
You are the debugging half of a documentation agent. The passages the agent looked up are
in the conversation above.

Work out the likeliest cause and give the reader the one change that tests it. Say what
they should see if you are right.

If what they described could be several things, name the most likely and say what would
tell it apart from the others.
""".strip(),
    ),
]


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    # Without a subagent the fast model answers everything itself and the skills mean
    # nothing: there would be nobody to hand the work to.
    config = await stream.define_agent(
        name="docs-agent",
        instructions=INSTRUCTIONS,
        llm="llm-fast",
        subagent="llm-smart",
        skills=SKILLS,
        knowledge=KNOWLEDGE,
    )
    print(f"talking to {config.name} ({config.id}), reading {KNOWLEDGE}\n")

    async with stream.TextSession(config_id=config.id) as session:
        print("ask something about the docs, or Ctrl-D to stop\n")
        while True:
            try:
                question = (await ask()).strip()
            except EOFError:
                print()
                return
            if not question:
                continue
            if question in ("quit", "exit"):
                return

            async for event in session.ask(question):
                show(event)
            print("\n")


async def ask() -> str:
    """Read a line without blocking the socket the answers arrive on."""
    print("> ", end="", flush=True)
    return await asyncio.to_thread(input)


def show(event: stream.TextEvent) -> None:
    """Print what the backend did, so the reader can see how the answer was arrived at."""
    if event.type == "delta":
        print(event.text, end="", flush=True)
    elif event.type == "answer":
        print()
    elif event.type == "looked_up":
        print(f"\n  [read {event.documents} passages about {event.query!r}]")
    elif event.type == "delegated":
        print(f"\n  [handed to {event.skill}: {event.text}]")
    elif event.type == "settled":
        if event.error:
            print(f"\n  [{event.skill} came back with nothing: {event.error}]")
        else:
            print(f"\n  [{event.skill} came back]")
    elif event.type == "error":
        print(f"\n  [error: {event.error}]", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
