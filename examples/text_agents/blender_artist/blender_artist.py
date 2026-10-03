import asyncio
import logging
import sys

from dotenv import load_dotenv
from vision_agents.core import Agent

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent that makes 3D renders with Blender.

Everything about Blender is in the agent's directory. `agent.yaml` gives the subagent the
router's Daytona sandbox and says how it is built: Blender as a Python module, installed
once on top of a slim Python image, with five minutes for a run. `skills/render.md` is
what the subagent writes its scene under. The render comes back as a file of run_code,
and the router uploads it to the conversation's channel and attaches it to the reply.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it. The
router needs DAYTONA_API_KEY, and Stream credentials to keep the conversation in Chat.
"""

REQUEST = "Render a low-poly lighthouse on a rocky island at dusk, its lamp lit."


async def main(request: str) -> None:
    agent = Agent(config="blender_artist")

    async with agent.chat():
        async for event in agent.ask(request):
            if event.type == "agent_speech_delta":
                print(event.text, end="", flush=True)
            elif event.type == "delegated":
                print(f"\n[handed to {event.skill}]")
            elif event.type == "task_settled":
                for file in event.files:
                    print(f"\n[{event.skill} made {file.name}: {file.url}]")
                if event.error:
                    print(f"\n[{event.skill} failed: {event.error}]")
            elif event.type == "error":
                print(f"[{event.error}]")
    print()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or REQUEST))
