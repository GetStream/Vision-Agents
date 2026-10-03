# Blender artist (text, with Blender in the sandbox)

A text agent that makes 3D renders. Ask for a picture: the agent hands it to a skill whose
subagent writes a Blender scene in Python and renders it in the router's Daytona sandbox.
The PNG is attached to the agent's reply in Chat.

```bash
cd examples/text_agents/blender_artist
uv sync
uv run blender_artist.py "a red teapot on a checkered floor"
```

The directory is the agent, and nothing in `blender_artist.py` sets up Blender:

- `agent.yaml` gives the subagent `sandbox: daytona`, and `sandbox_options` says how the
  sandbox is built: `setup` installs Blender's libraries and `bpy==5.2.2` on a slim Python
  3.13 image, `timeout` allows five minutes for a run, and `cpu` and `memory_gb` size it.
  Daytona keeps the built image, so only the first sandbox waits for the build, which takes
  a few minutes.
- `skills/render.md` is what the subagent writes under: a starting program that frames the
  scene and renders it with Cycles, and the instruction to pass
  `files=["/tmp/render.png"]` to `run_code`. Its `deadline: 20m` covers the first build.
- `instructions.md` is the conversation model, which describes the scene to the skill and
  says what came back.

The router downloads the file `run_code` names, uploads it to the conversation's channel,
and attaches it to the reply that settles the work, so it is still there when the
conversation is reopened. `task_settled` carries the file's URL as well, which is what this
script prints.

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`, plus the Stream app's `STREAM_API_KEY` and
`STREAM_API_SECRET`. The router needs `DAYTONA_API_KEY` and the same Stream credentials,
since the render goes to the conversation's channel.
