# Development

## Local backend

The router, Postgres and Redis run from `compose.yaml` at the repo root. Credentials come
from the repo-root `.env` (copy `.env.example` if you have none).

```bash
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

The price list lives in that Postgres, seeded from `acceleration/prices/vendor.json` on the
first start; later starts only add items it has never priced. After changing a rate in
`vendor.json`, put the file's list back with
`cd acceleration && go run ./cmd/router prices reload --development`. It refuses a database
that is not on this machine or compose's, but a port-forward on localhost to a remote one
passes that check, so read the database and address it prints first, and start with
`--dry-run`. See Prices in `acceleration/README.md`.

The dashboard is Volt (`GetStream/volt-dashboard`), a sibling checkout. See the `dashboard`
skill and `acceleration/development_skills/dashboard_skill.md`.

## Pointing an SDK at a router

The SDKs go to production, `https://accelerate.gcp.stream-io-api.com`, when nothing says
otherwise. Staging is `https://accelerate-staging.gcp.stream-io-api.com`, and the load
balancer in `GetStream/chat`'s `infra/dns/gcp-stream-io-api-com` tells them apart by hostname. The server SDKs (Python, JS, Go, Ruby, PHP, .NET and Rust) read the same variables
to go anywhere else:

| Variable                           | What it does                                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `STREAM_ACCELERATION_URL`          | The router's base URL. Unset is production.                                                    |
| `STREAM_ACCELERATION_CUSTOMER_ID`  | Who is billed, on a router with no proxy in front of it. Setting it turns authentication off.  |
| `STREAM_API_KEY`                   | The Stream app a router behind the proxy is reached as.                                        |
| `STREAM_API_SECRET`                | That app's secret, which signs the token the proxy checks.                                     |
| `STREAM_ACCELERATION_AUTHENTICATE` | `1` sends the key and a token through the proxy. On by itself only for production.            |

Local, the router from `compose.yaml` above, which trusts the customer header and has no proxy:

```
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

Staging, `accelerate-staging` in `GetStream/chat`'s `infra/shards/us-east1-acceleration`, sits
behind the proxy like production does, so it takes an app's key and secret:

```
STREAM_ACCELERATION_URL=https://accelerate-staging.gcp.stream-io-api.com
STREAM_ACCELERATION_AUTHENTICATE=1
STREAM_API_KEY=...
STREAM_API_SECRET=...
```

Without `STREAM_ACCELERATION_AUTHENTICATE=1` every SDK but Go sends staging its key the way a
router with no proxy reads it, and the proxy refuses the request. Go authenticates whenever
no customer id is set.

Production takes only the key and secret. Leave `STREAM_ACCELERATION_CUSTOMER_ID` unset
there: a customer id left over from local turns authentication off.

The Kotlin, Swift and Dart SDKs run on a phone and read no environment: pass the URL to
their `Backend` (`Backend(url = ...)`, `Backend(apiKey:url:)`, `Backend(url: ...)`). The JS
SDK in a browser has no environment either, so pass `url` there too.

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
`decision_model` (what each router supports), `stt` / `tts` (adding a provider),
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
| `internal/<modality>` | The contract per modality: `llm`, `stt`, `tts`, `sts`, `search`, `decisionmodel`, `imagegen`. Providers live in subfolders |
| `internal/<modality>router` | Registers the providers of that modality and runs sessions |
| `internal/agent` | The conversation loop: transcribe, answer, speak, barge-in. `streamedge/` is the WebRTC transport |
| `internal/harness`, `internal/session` | Skills, delegation and tools around the model; agent session lifecycle |
| `internal/api` | HTTP handlers, declared in Go with Huma; `api/openapi.yaml` is rendered from them |
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
