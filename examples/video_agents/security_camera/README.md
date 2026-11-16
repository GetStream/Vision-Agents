# Security camera

A voice agent watching a door camera. It notices when someone arrives or leaves and when a
package appears or is picked up, names the people it was told to remember, and can say
what happened while you were away.

The Go router handles speech. This Python process joins the same call as a video worker
and runs two `stream.DetectionProcessor`s on the camera, both through the router's
`POST /v1/detection`:

| Processor | Target            | What it finds                                    |
| --------- | ----------------- | ------------------------------------------------ |
| `objects` | `detection-open`  | people and packages, by asking Moondream for them |
| `faces`   | `detection-faces` | faces, each with an embedding, from a Roboflow Workflow |

Frames that barely changed are never sent, and an unchanged scene is looked at again every
10 seconds, so a camera on an empty porch costs little. Boxes are followed across frames,
and what comes and goes is written to an activity log and, a few seconds later, told to the
agent as a `[camera]` message so it speaks up on its own.

```
agent.yaml           models and video selection; synced to the router on join
instructions.md      what the agent is told
skills/vision.md     the skill that looks at the frames around a moment
security_camera.py   the video worker: detectors, faces, activity log, tools
faces.json           who remember_me enrolled; written on first use, not committed
```

The agent's tools:

- `get_video_state`: what each processor sees now, with names on the faces it knows.
- `get_activity_log`: arrivals, departures, packages appearing and being picked up, with
  times. A package that stays gone for 3 seconds while someone was at the door within the
  last 10 is logged as picked up by them, by name when their face was recognised.
- `remember_me(name)`: stores the largest face in view under `name`.
- The `vision` skill: looks at the retained frames, for what the detectors cannot say or to
  check a logged event against the frames around it.

## Faces

`detection-faces` is `roboflow/faces`, an inline Roboflow Workflow the router sends with
every request: Roboflow's `face-detection-mik1i/18` model, a crop of every face, and CLIP
on each crop. Nothing has to be set up in Roboflow. Matching is cosine similarity against
`faces.json`, at 0.85 (`MATCH_THRESHOLD`): two photos of one person scored 0.92 and two
people 0.76. CLIP is not a face recognition model, so treat a name as a good guess rather
than an identity.

To use a Workflow of your own instead, with a better embedding model, point the faces
processor at `roboflow/workflow` and name it:

```python
stream.DetectionProcessor(
    target="roboflow/workflow",
    embeddings=True,
    overwrites={"roboflow": {"workspace": "my-workspace", "workflow_id": "faces"}},
    ...
)
```

It has to output `predictions` and `embeddings`, one embedding per prediction.

## Prerequisites

A running acceleration router from the repo root, rebuilt so it has detection:

```bash
docker compose up -d --build router
```

The router reads the provider keys from the repo-root `.env`:

```
MOONDREAM_API_KEY=your_moondream_key
ROBOFLOW_API_KEY=your_roboflow_key
STREAM_API_KEY=your_stream_key
STREAM_API_SECRET=your_stream_secret
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

plus the keys of its speech and LLM defaults (`GOOGLE_API_KEY`, `OPENAI_API_KEY` for
`vlm`, `ELEVENLABS_API_KEY`). `ULTRALYTICS_API_KEY` is optional: `detection-fast` falls
back to Ultralytics' shared YOLO endpoint without one.

Get a Moondream key at [moondream.ai](https://moondream.ai) and a Roboflow key from
Roboflow, Settings, API.

## Run

```bash
cd examples/video_agents/security_camera
uv sync
uv run security_camera.py run
```

`run` prints a call URL and opens the demo page on it. Turn the camera on and try:

1. Put a box in view. The agent says a package appeared.
2. Face the camera and say "remember me as Sam". Leave, come back: it greets Sam.
3. Pick the box up and walk off. Ask "what happened?": the package was picked up by Sam.
4. Ask "what was I wearing when I took it?": the vision skill looks at the frames.

Detection costs about $0.0005 a look for `objects` (Moondream bills each class) and
$0.0015 for `faces`, and `get_video_state` reads what was last found without a call.
