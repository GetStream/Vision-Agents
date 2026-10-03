import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.core.mcp import MCPServerLocal
from vision_agents.plugins import stream

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent that makes 3D renders with Blender.

Blender is reached through MCP: `blender_mcp.py` is a stdio MCP server, started here as a
local one, whose `render` tool runs a bpy script in a Daytona sandbox and saves the PNG
under `renders/`. The tool belongs to this process rather than the backend, so the session
sends each render back here to run, the way it does any caller-owned tool.

A first render waits for Daytona to build the Blender image, which takes minutes, so the
model is allowed far longer than the default to wait on a tool.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it.
Rendering is Daytona, so this process needs DAYTONA_API_KEY.
"""

REQUEST = "Render a low-poly lighthouse on a rocky island at dusk, its lamp lit."

DAYTONA_ENV = ("DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET")


async def main(request: str) -> None:
    blender = MCPServerLocal(
        command=f"{sys.executable} {Path(__file__).with_name('blender_mcp.py')}",
        env={name: os.environ[name] for name in DAYTONA_ENV if name in os.environ},
        session_timeout=1800,
    )
    agent = Agent(
        config="blender_artist",
        llm=stream.Accelerated(config="blender_artist", tool_timeout=1200),
        mcp_servers=[blender],
    )

    async with agent.chat():
        async for event in agent.ask(request):
            if event.type == "agent_speech_delta":
                print(event.text, end="", flush=True)
            elif event.type == "error":
                print(f"[{event.error}]")
    print()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or REQUEST))
