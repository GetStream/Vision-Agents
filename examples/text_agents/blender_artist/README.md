# Blender artist (text, with Blender MCP)

A text agent that makes 3D renders. Ask for a picture and it writes a Blender scene in
Python, renders it, and tells you where the PNG is.

```bash
cd examples/text_agents/blender_artist
uv sync
uv run blender_artist.py "a red teapot on a checkered floor"
```

Renders are saved under `renders/`.

## How it fits together

- `blender_mcp.py` is a stdio MCP server with one tool, `render`. Any MCP client can start
  it; this agent does with `MCPServerLocal`.
- Blender never runs locally. The server boots one Daytona sandbox from an image with
  `bpy` 5.2 installed, runs `render.py` there on the model's scene script, and
  downloads the PNG. The sandbox is reused for every render and deleted when the server
  exits. If the server can't delete it, the sandbox stops itself after 15 idle minutes.
- `render.py` renders with Cycles on the CPU. It adds a camera, a light and a world if the
  script leaves them out. It runs anywhere `bpy` is installed, so
  `python render.py scene.py out.png 960 540 32` tries a scene without Daytona.

The tool belongs to this process, not to the backend, so the session sends each render
back here to run. The backend's `sandbox: daytona` is not used: it runs short Python for
the subagent and only returns text, while a render needs Blender's image, minutes of
build time on first use, and a PNG back.

The first render waits for Daytona to build the Blender image, which takes a few minutes.
Daytona caches the image after that, and later sandboxes start in seconds. The server
starts building as soon as the agent connects, and the agent gives a tool 20 minutes
instead of the default 30 seconds.

## Needs

- A router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
  `STREAM_ACCELERATION_CUSTOMER_ID`, plus the Stream app's `STREAM_API_KEY` and
  `STREAM_API_SECRET`.
- `DAYTONA_API_KEY` in this process's environment, not the router's. `DAYTONA_API_URL`
  and `DAYTONA_TARGET` are passed through too when set.
