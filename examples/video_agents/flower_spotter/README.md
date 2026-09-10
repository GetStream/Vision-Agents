# Flower spotter

A voice agent that looks through an iOS camera. The Go router handles speech;
this Python process joins the same call as a video worker, streams the camera
to Roboflow Serverless Video Streaming, and publishes the frames back with
boxes drawn on.

The phone shows its own camera with the annotated track inset. The model does
**not** see those frames. It only gets detection labels (and boxes/counts)
when it calls `get_video_state`.

```
agent.yaml           name and description; pushed to the router on join
instructions.md      what the agent is told
flower_spotter.py    joins as the video worker
app/                 iOS: lists the live call, camera on, annotated video
```

The default model is `rfdetr-nano`, a serverless COCO detector (person, cup,
laptop, cat). That is the office smoke test. For flowers, pass `workflow_id=`
and `workspace=` instead of `model_id=` and rewrite `instructions.md` so it
still answers only from `get_video_state`.

Streaming is billed per hour while the Roboflow WebRTC session is open. The
red hang-up button in the app ends the session and closes it.

## Prerequisites

A running acceleration router from the repo root:

```bash
docker compose up --build
```

That serves the router on `:8080`. See [acceleration/README.md](../../../acceleration/README.md).

Credentials live in the repo-root `.env`. This example needs:

```
STREAM_API_KEY=your_stream_key
STREAM_API_SECRET=your_stream_secret
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
ROBOFLOW_API_KEY=your_roboflow_key
```

`STREAM_ACCELERATION_CUSTOMER_ID` is often missing from `.env`. It must be
`examples` so it matches `Demo.customerID` in the app and
`NEXT_PUBLIC_CUSTOMER_ID` in `compose.yaml`. Export it if it is not in the
file.

The router also needs a key for whichever LLM it routes the config to,
`GOOGLE_API_KEY` by default. Get a Roboflow key from Roboflow → Settings → API.

This example has its own venv. The Roboflow `webrtc` extra needs NumPy 2,
which the workspace pins away.

## Run the video worker

```bash
cd examples/video_agents/flower_spotter
uv sync
export STREAM_ACCELERATION_CUSTOMER_ID=examples
uv run flower_spotter.py run --no-demo
```

`--no-demo` skips the browser UI; the phone is the client. Without it, `run`
also opens the dashboard on the same call. `--call-id` joins a named call
instead of a new one.

The process prints a call id and waits. Leave it running. It exits when the
call ends (hang-up, not navigating back).

## Run the iOS app

Open `app/FlowerSpotter.xcodeproj` on a **physical device**. The camera is the
point; the simulator can reach the router but cannot usefully point a camera.

Two constants in `app/FlowerSpotter/Demo.swift`:

| Constant | Simulator | Physical device |
| --- | --- | --- |
| `routerURL` | `http://localhost:8080` | Mac LAN address, `http://$(ipconfig getifaddr en0):8080` |
| `customerID` | `examples` | `examples` |

`NSAllowsLocalNetworking` in `Info.plist` lets plain HTTP through to that
address. Allow Local Network when iOS asks.

Rebuild after changing `Demo.routerURL`. Pull to refresh, tap the live call,
point the camera, and ask what it sees.

Navigating back leaves the phone's call and keeps the Python session. The red
hang-up button ends the session.

If the list is empty, the worker is not running. If it shows a call that is
already dead, a killed `--no-demo` worker left a session with no `ended_at`;
close it with `DELETE /v1/agents/sessions/{id}` and `X-Customer-Id: examples`.
