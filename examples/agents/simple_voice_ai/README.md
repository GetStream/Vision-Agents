# Simple voice AI (accelerated)

A voice agent with nothing in the pipeline running in Python. `instructions.md` says what
the agent is, and the four targets say who does the work:

| Modality     | Target                              |
| ------------ | ----------------------------------- |
| Transcribe   | `gemini/gemini-3.5-transcribe-live` |
| Answer       | `gemini/gemini-3.5-flash-lite`      |
| Speak        | `elevenlabs/eleven_v3_conversational` |
| Think        | `openai/gpt-5.6-sol`                |

The agent has no `skills/` directory, so it gets the built-in `think`, `recall` and
`explain`. Work handed to any of them runs on Sol while the conversation carries on.

`eleven_v3_conversational` performs bracketed directions such as `[laughs]`, and the
backend tells the model so, which is why the instructions here say nothing about them. It
is around a second to first audio rather than flash's 75ms; swap in
`elevenlabs/eleven_flash_v2_5` if latency matters more than performance.

## Prerequisites

A running acceleration router: see [acceleration/README.md](../../../acceleration/README.md).

- `ROUTER_POSTGRES_DSN`, since a stored agent config is a row. Without it the sync is
  refused rather than half-applied.
- `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY` and `OPENAI_API_KEY`. Provider credentials live
  with the router, not here.
- `TAVILY_API_KEY`, optionally, which is what lets the agent answer about traffic, weather
  or anything else that depends on today. Without it there is no `search` tool and the
  agent can only say it cannot check: Sol reasons well but knows nothing about this
  morning.

## Run

```bash
cd examples/agents/simple_voice_ai
uv sync
uv run simple_voice_ai.py
```

Needs a `.env` with:

```
STREAM_API_KEY=your_stream_key
STREAM_API_SECRET=your_stream_secret
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

The first run pushes `instructions.md` to the router; a second run with the same file does
nothing.
