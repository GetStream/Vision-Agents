"""
A Blender MCP server whose Blender runs in a Daytona sandbox.

It speaks MCP over stdio, so the agent starts it as a local server, and any other MCP
client can too. Blender itself never runs here: the first render, or the server starting,
boots one Daytona sandbox with Blender installed, and every render after that reuses it
until the server exits.

Needs DAYTONA_API_KEY.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    DaytonaError,
    Image,
    Resources,
)
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp import Image as Picture

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
RENDERS = HERE / "renders"

# bpy is Blender as a Python module. Its wheel brings Blender, and these are the system
# libraries it links against that a slim Debian does not have.
BLENDER = (
    Image.debian_slim("3.13")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "libgl1 libx11-6 libxext6 libxfixes3 libxi6 libxkbcommon0 libxrender1 "
        "libxxf86vm1 libsm6 libice6 libxt6 && rm -rf /var/lib/apt/lists/*"
    )
    .pip_install("bpy==5.2.2")
)

DRIVER = "/tmp/render.py"
SCENE = "/tmp/scene.py"
OUTPUT = "/tmp/render.png"

# Building the image the first time takes minutes; Daytona keeps it after that.
CREATE_TIMEOUT = 900
RENDER_TIMEOUT = 300
# A sandbox the server could not delete on the way out stops itself and goes away.
IDLE_MINUTES = 15


class RenderError(Exception):
    """Blender ran the scene and it did not render."""


class Studio:
    """One Daytona sandbox with Blender in it, shared by every render."""

    def __init__(self) -> None:
        self._daytona = AsyncDaytona()
        self._sandbox: Optional[AsyncSandbox] = None
        self._lock = asyncio.Lock()

    async def warm(self) -> None:
        """Boot the sandbox ahead of the first render, which would otherwise wait for it."""
        try:
            await self._ready()
        except DaytonaError:
            logger.exception("could not start the Blender sandbox")

    async def render(self, script: str, width: int, height: int, samples: int) -> bytes:
        """Run a scene script in Blender and return the PNG it rendered."""
        sandbox = await self._ready()
        async with self._lock:
            await sandbox.fs.upload_file(script.encode(), SCENE)
            response = await sandbox.process.exec(
                f"python {DRIVER} {SCENE} {OUTPUT} {width} {height} {samples}",
                timeout=RENDER_TIMEOUT,
            )
            if response.exit_code != 0:
                raise RenderError(response.result[-4000:])
            return await sandbox.fs.download_file(OUTPUT)

    async def close(self) -> None:
        """Delete the sandbox, if one was started."""
        try:
            if self._sandbox is not None:
                await self._sandbox.delete()
        finally:
            await self._daytona.close()

    async def _ready(self) -> AsyncSandbox:
        async with self._lock:
            if self._sandbox is None:
                started = time.monotonic()
                sandbox = await self._daytona.create(
                    CreateSandboxFromImageParams(
                        image=BLENDER,
                        resources=Resources(cpu=2, memory=4, disk=10),
                        auto_stop_interval=IDLE_MINUTES,
                        ephemeral=True,
                    ),
                    timeout=CREATE_TIMEOUT,
                )
                await sandbox.fs.upload_file((HERE / "render.py").read_bytes(), DRIVER)
                self._sandbox = sandbox
                logger.info(
                    "Blender sandbox %s ready in %.0fs",
                    sandbox.id,
                    time.monotonic() - started,
                )
            return self._sandbox


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[Studio]:
    studio = Studio()
    warming = asyncio.create_task(studio.warm())
    try:
        yield studio
    finally:
        warming.cancel()
        await studio.close()


mcp = FastMCP(
    "blender",
    instructions="Renders 3D scenes with Blender, from Python that builds them with bpy.",
    lifespan=lifespan,
)


@mcp.tool(structured_output=False)
async def render(
    script: str,
    ctx: Context,
    width: int = 960,
    height: int = 540,
    samples: int = 32,
) -> list[str | Picture]:
    """Render a 3D scene with Blender and save it as a PNG.

    The script is Python run inside Blender (bpy 5.2) on an empty scene, so it builds
    everything: meshes, materials, lights and a camera. Without a camera one is placed
    looking at the scene; without a light a sun is added. Do not set render settings or
    call bpy.ops.render; that happens after the script.

    Args:
        script: Python using bpy that builds the scene.
        width: Image width in pixels, at most 1920.
        height: Image height in pixels, at most 1080.
        samples: Cycles samples. 32 is quick; 128 is cleaner and slower.
    """
    studio: Studio = ctx.request_context.lifespan_context
    started = time.monotonic()
    png = await studio.render(
        script, min(width, 1920), min(height, 1080), max(1, min(samples, 512))
    )
    RENDERS.mkdir(exist_ok=True)
    path = RENDERS / f"render-{time.strftime('%Y%m%d-%H%M%S')}.png"
    path.write_bytes(png)
    return [
        f"Rendered in {time.monotonic() - started:.0f}s and saved to {path}",
        Picture(data=png, format="png"),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
