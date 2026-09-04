# Simple voice AI (accelerated)

A voice agent with nothing in the pipeline running in Python. `instructions.md` says what
the agent is, and it names no models at all: a config that says nothing about who does the
work gets the router's defaults.

| Modality   | Default target               |
| ---------- | ---------------------------- |
| Transcribe | `en-low-latency`             |
| Answer     | `llm-fast`                   |
| Speak      | `en-low-latency`             |
| Think      | `multilingual-high-accuracy` |

Each of those is a capability rather than a model, so the router picks inside the tier by
live health and a degraded provider drops down the list. Set `stt`, `tts`, `llm` or
`subagent` on the config to pin any of them to one provider.

The agent has no `skills/` directory, so it gets the built-in `think`, `recall` and
`explain`. Work handed to any of them runs on the thinking model while the conversation
carries on.

## Prerequisites

A running acceleration router: see [acceleration/README.md](../../../acceleration/README.md).

- `ROUTER_POSTGRES_DSN`, since a stored agent config is a row. Without it the sync is
  refused rather than half-applied.
- A key for whichever providers the router routes each default to, `GOOGLE_API_KEY`,
  `ELEVENLABS_API_KEY` and `OPENAI_API_KEY` among them. Provider credentials live with the
  router, not here.
- `TAVILY_API_KEY`, optionally, which is what lets the agent answer about traffic, weather
  or anything else that depends on today. Without it there is no `search` tool and the
  agent can only say it cannot check: a thinking model reasons well but knows nothing
  about this morning.

## Run

```bash
cd examples/agents/simple_voice_ai
uv sync
uv run simple_voice_ai.py run
```

`run` joins one call and opens the demo UI on it, which is what to talk into. `--call-id`
joins a call by name rather than a new one, `--no-demo` leaves the browser alone, and
`serve` instead starts the HTTP server that sends an agent to whichever call asks for one.

Needs a `.env` with:

```
STREAM_API_KEY=your_stream_key
STREAM_API_SECRET=your_stream_secret
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

The first run pushes `instructions.md` to the router; a second run with the same file does
nothing.
