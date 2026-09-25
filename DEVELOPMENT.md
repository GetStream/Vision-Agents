# Development

## Local backend

The router, Postgres and Redis run from `compose.yaml` at the repo root. Credentials come
from the repo-root `.env` (copy `.env.example` if you have none).

```bash
ssh-add -l                            # needs a key with access to GetStream/getstream-go-webrtc
docker compose up -d --build router   # router on :8080, Postgres on :55432, Redis on :56379
curl http://localhost:8080/health
```

Rebuild after backend changes: `docker compose up -d --build router`. Logs:
`docker compose logs -f router`.

Without Docker for the router itself (Postgres and Redis still from compose):

```bash
docker compose up -d postgres redis
cd acceleration
ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router?sslmode=disable' \
ROUTER_REDIS_ADDR=localhost:56379 go run ./cmd/router
```

The dashboard is Volt (`GetStream/volt-dashboard`), a sibling checkout. See the `dashboard`
skill and `acceleration/development_skills/dashboard_skill.md`.

## Simple voice AI example

With the router up, add to `.env`:

```
STREAM_API_KEY=...
STREAM_API_SECRET=...
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

```bash
cd examples/voice_agents/simple_voice_ai
uv sync
uv run simple_voice_ai.py run
```

It opens a demo call in the browser; talk into it. More in the example's `README.md`.

## Skills

In `.claude/skills/`. Internal only:

- `docs`: editing the agents docs on getstream.io
- `dashboard`: working on Volt and wiring router changes into it
- `deploy`: shipping `cmd/router` to Stream's hosted environments

Also useful: `router-llm`, `router-stt`, `router-tts`, `router-sts`, `router-search`,
`router-lcm` (what each router supports), `stt` / `tts` (adding a provider),
`new_models` / `integrate_new_models`, the `sdk-*` skills per language, `commit` and `pr`.

## Specs

`.factory/` has the factory specs: `overview.md`, the sprint plans, `features/` and `evals/`.

## Architecture (`acceleration/`)

A Go service that routes STT, TTS, LLM, speech-to-speech, search and image generation
across providers behind one API, plus an agent that joins a call and talks.

| Path | What it is |
| --- | --- |
| `api/openapi.yaml` | Source of truth for the HTTP API; clients are generated from it |
| `cmd/router` | The server. Other `cmd/*` are small CLIs (`agent`, `chat`, `say`, `phone`, ...) |
| `internal/routing` | Modality-agnostic core: config, registry, selection, failover, stats. `router.yaml` lists models and tiers |
| `internal/<modality>` | The contract per modality: `llm`, `stt`, `tts`, `sts`, `search`, `lcm`, `imagegen`. Providers live in subfolders |
| `internal/<modality>router` | Registers the providers of that modality and runs sessions |
| `internal/agent` | The conversation loop: transcribe, answer, speak, barge-in. `streamedge/` is the WebRTC transport |
| `internal/harness`, `internal/session` | Skills, delegation and tools around the model; agent session lifecycle |
| `internal/api` | HTTP handlers (`generated.go` from the spec) |
| `internal/store`, `migrations/` | Postgres via bun, goose migrations |
| `internal/live` | Redis: provider health and live counters |
| `internal/phone`, `knowledge`, `memory`, `chatlog` | Telephony, knowledge bases, memory, chat transcripts |
| `deploy/` | Truss deployments of self-hosted models on Baseten |

Full layout in `acceleration/README.md`.

## Related projects

- `chat/infra` (GetStream/chat): infrastructure for the hosted router
- `getstream.io`: the public docs
- `artemis-impl`: the support AI
- TODO: document `itheqa` and `athena`
