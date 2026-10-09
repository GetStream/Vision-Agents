# Model router and agent backend

This Go service runs voice and text agents. It routes model requests, manages
conversation history and tools, and joins Stream calls on an agent's behalf.
You can run one voice agent directly, or run the HTTP service for the
[Go](../sdks/go/README.md), [Python](../plugins/stream/README.md) and
[JavaScript](../sdks/js/README.md) clients.

- [Try a voice agent](#try-a-voice-agent)
- [Run the API locally](#run-the-api-locally)
- [Choose models](#choose-models)
- [How voice turns work](#how-voice-turns-work)
- [Configure a deployment](#configure-a-deployment)
- [Sessions and history](#sessions-and-history)
- [Debug and develop](#debug-and-develop)

## Try a voice agent

Install Go 1.27 or newer. The audio path builds in pure Go; no system Opus or
SoXR libraries are required.

Put credentials in the repository-root `.env`. This example uses Deepgram for
transcription, OpenAI for replies and Cartesia for speech:

```dotenv
STREAM_API_KEY=...
STREAM_API_SECRET=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
```

From the repository root:

```bash
cd acceleration
go run ./cmd/agent -call my-call \
  -stt deepgram/flux-general-en \
  -llm openai/gpt-6.1-sol \
  -tts cartesia/sonic-preview
```

The agent joins the call and opens a browser link to the same call. Allow
microphone access and speak. Use `-demo=false` if another client already joins
the call, and Ctrl-C to stop the agent. This command runs the model routers in
process: it does not need the HTTP service, Postgres or Redis.

The nearest parent `.env` is loaded without replacing variables already set in
the shell. Run `go run ./cmd/agent -h` for all flags. Backchannels and idle
check-ins are off by default; `-backchannel` and `-check-in` enable them.

The other commands read credentials from the process environment. Export your
`.env` before checking individual providers:

```bash
set -a
source ../.env
set +a
go run ./cmd/chat -target openai/gpt-6.1-sol -text "Say hello in one sentence."
go run ./cmd/say -target cartesia/sonic-preview -text "Hello from the router."
```

## Run the API locally

From the repository root, start the database and Redis:

```bash
docker compose up -d postgres redis
```

Then start the router in another terminal:

```bash
cd acceleration
set -a
source ../.env
set +a
ROUTER_AUTH_MODE=noauth go run ./cmd/router
```

The local configuration connects to Postgres on `localhost:55432` and Redis on
`localhost:56379`. The router applies database migrations at startup and serves
HTTP on port 8080. Check it with:

```bash
curl -fsS http://localhost:8080/health
```

`noauth` is for local development. Requests identify their customer with
`X-Customer-Id`; no credential is checked. Production authentication is described
[below](#authentication).

Create a voice session using the same credentials as the standalone example:

```bash
curl -sS http://localhost:8080/v1/agents/sessions \
  -H 'X-Customer-Id: demo' -H 'Content-Type: application/json' \
  -d '{
    "call_id": "my-call",
    "instructions": "Keep your replies short.",
    "stt": "deepgram/flux-general-en",
    "llm": "openai/gpt-6.1-sol",
    "tts": "cartesia/sonic-preview"
  }'
```

This starts an agent in the call; your client still needs to join that call.
For a complete client with a browser join link, use the
[Go voice example](../sdks/go/README.md#in-a-call) or
[Python example](../examples/voice_agents/simple_voice_ai).

The [OpenAPI specification](api/openapi.yaml) describes all request fields,
responses and authentication requirements. Model requests and agent sessions
share the same routing and billing code.

## Choose models

Each modality accepts either a concrete `provider/model` or a routing target:

| Target | Purpose |
| --- | --- |
| `deepgram/flux-general-en` | Pin transcription to one provider and model |
| `cartesia/sonic-preview` | Pin speech synthesis |
| `openai/gpt-6.1-sol` | Pin the reply model |
| `en-low-latency` | Choose from the configured low-latency English models for that modality |
| `llm-fast` | Choose from the fast reply models |

Targets filter candidates by capability and available credentials, then rank
and fail over between them. A pinned model has no alternative candidate. See
[router.yaml](internal/routing/router.yaml) for the actual target definitions,
model capabilities and configured prices. Set `ROUTER_CONFIG` to use a different
routing file; it is separate from the deployment configuration.

You only need credentials for the providers you use. Common variables are
`DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `ELEVENLABS_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_API_KEY`, `INWORLD_API_KEY`, `FISH_API_KEY` and `BASETEN_API_KEY`.
Self-hosted providers use their endpoint settings, such as `PARAKEET_WS_URL`.

## How voice turns work

A cascaded voice agent transcribes incoming audio, decides when the caller has
finished, generates a reply, then speaks it:

```text
caller audio → transcription → turn decision → reply model → speech synthesis → call
                                  ↑
                         acoustic end-of-turn score
```

Transcription can change while the caller is speaking. The agent starts an early
reply after the words have stayed stable for 60 ms and the audio has been quiet
for 120 ms. That preview cannot speak or execute tools until its turn is
accepted. New words replace it; discarded previews still incur model usage.
There are at most three early previews per utterance.

### Acoustic and semantic decisions

By default, both binaries use the hosted EU end-of-turn (EOT) scorer in `primary`
mode. It receives up to 16 seconds of trailing caller audio as 16 kHz mono
PCM16LE over HTTPS. The hosted service requires no EOT credentials.

For an eligible quiet-floor turn, a score at or above `0.5` accepts the ending;
a lower score waits for more audio. This score answers whether the speaker has
finished. A semantic check separately determines whether the words address the
agent and can withdraw a reply before it is heard. Interruptions and other
ineligible candidates continue through the semantic controller.

A transient scoring failure gets at most three attempts within a shared
one-second budget, then falls back to the semantic controller. The hosted demo
has limited capacity and no availability guarantee. Its transient startup
preflight failures warn and allow startup; permanent failures stop the demo.

| Environment variable | Effect |
| --- | --- |
| `ROUTER_EOT_URL` | Unset uses the hosted demo. Empty disables acoustic scoring. A private URL selects your own `/v1/eot` service. |
| `ROUTER_EOT_MODE` | `primary` uses EOT for eligible turn endings. `gate` requires the semantic decision as well. The hosted default is `primary`; a private URL defaults to `gate`. |
| `ROUTER_EOT_THRESHOLD` | Acceptance probability, default `0.5`, range `[0, 1]`. |
| `ROUTER_EOT_ID_TOKEN_FILE` | Identity-token file for a private service. Otherwise private HTTPS endpoints use Google Application Default Credentials. Never used for the hosted demo. |

For example, to use only transcript and semantic turn detection:

```bash
ROUTER_EOT_URL= go run ./cmd/agent -call my-call
```

### Why a ready reply may wait

Before publishing the first reply audio, the agent checks how long the caller
has been quiet. The default is 700 ms, shortened to 300 ms when the accepted
acoustic score is at least `0.9`. Continuous noise can hold ready audio for at
most one second. New caller speech restarts the silence measurement; new words
can cancel a reply that has not been heard.

For example, if the caller stops at **0 ms** and reply audio is ready at
**250 ms**, a confident ending can speak at **300 ms**: the hold adds **50 ms**.
An ordinary ending waits until **700 ms**, adding **450 ms**. If audio is only
ready at 800 ms and the caller stayed quiet, neither setting adds a wait.
Greetings and backchannels bypass this hold.

| Router environment variable | `cmd/agent` flag | Default |
| --- | --- | --- |
| `ROUTER_REPLY_SILENCE` | `-reply-silence` | `700ms`; `0` disables |
| `ROUTER_REPLY_SILENCE_MAX` | `-reply-silence-max` | `1s` maximum hold once audio is ready |
| `ROUTER_REPLY_SILENCE_CONFIDENT` | `-reply-silence-confident` | `300ms` |
| `ROUTER_REPLY_CONFIDENT_SCORE` | `-reply-confident-score` | `0.9`; `0` disables the shorter silence |
| `ROUTER_PREVIEW_DEBOUNCE` | `-preview-debounce` | `60ms`; `0` waits for the settled candidate |
| `ROUTER_PREVIEW_QUIET` | `-preview-quiet` | `120ms`; `0` checks only transcript stability |
| `ROUTER_REPLY_HEDGE` | `-reply-hedge` | `1200ms`; `0` disables |

Use environment variables for `cmd/router` and flags for these `cmd/agent`
settings. `ROUTER_SPECULATIVE_REPLIES=false` disables reply previews in the router.
Hedging starts one alternative reply request if the first has produced neither
text nor a tool call by the deadline; the first to respond wins.

When the caller interrupts, queued playback is cleared and stale speech is
cancelled. Tool work follows its cancellation policy. Text sessions and native
speech-to-speech sessions do not use the cascaded EOT and first-audio gates.

## Configure a deployment

```bash
go run ./cmd/router --config /etc/router.yaml
```

Settings load in this order: built-in defaults, `ROUTER_ENV`'s embedded YAML,
an optional `--config`/`ROUTER_CONFIG_FILE`, then environment overrides.
`ROUTER_ENV` is `local` by default; `staging` and `testing` have their own files.
The testing profile protects its test database by overriding environment values.
See [config.go](internal/config/config.go) for all fields, defaults and variable
names, and [local.yaml](internal/config/local.yaml) for the local service ports.

| Setting | Use |
| --- | --- |
| `ROUTER_ADDR` | Listen address, default `:8080` |
| `ROUTER_PUBLIC_URL` | Externally reachable router URL, including OAuth callbacks |
| `ROUTER_POSTGRES_DSN` | Database for configuration, request records and statistics |
| `ROUTER_REDIS_ADDR` | Provider health, daily limits, workers and communication between nodes |
| `ROUTER_CORS_ORIGINS` | Comma-separated browser origins allowed for HTTP and WebSocket access |
| `ROUTER_TRUSTED_PROXIES` | Proxy CIDRs trusted to supply `X-Forwarded-For` |
| `ROUTER_LOG_LEVEL` | `debug`, `info`, `warn` or `error` |
| `ROUTER_VOICES_BUCKET_URL` | Object storage for custom voice recordings, e.g. `s3://voices?region=eu-west-1` |
| `ROUTER_CONNECTORS_ENABLED` | Enable connectors; requires the credential keyring in every auth mode |

### Authentication

| Mode | Credentials and intended deployment |
| --- | --- |
| `api_key` (default) | Requires Postgres and `ROUTER_AUTH_KEK`. Send `X-Api-Key` and an HS256 bearer token signed with the corresponding secret. |
| `proxy` | Trusts a gateway's `X-Stream-App-Id`, `X-Stream-Organization-Id`, `X-Stream-User-Id` and `Stream-Auth-Type`. Restrict access to that gateway and have it overwrite these headers. |
| `noauth` | Local development. Reads `X-Customer-Id` and treats the caller as that customer's backend. |
| `custom` | For embedding with an `auth.Authenticator`; unavailable in the stock binary. |

Bootstrap an API key beside the database:

```bash
go run ./cmd/router keys create --org Acme --app-name "Acme production" --key-name default
```

The secret is printed once. Keep `ROUTER_AUTH_KEK` outside the database and stable
across restarts; changing it makes existing sealed credentials unreadable. Quote
its value in `.env` so Compose does not expand `$`. Connector key rotation uses
`ROUTER_AUTH_KEK_V<n>` and `ROUTER_AUTH_KEK_VERSION`; retain older versions until
their stored credentials have been rewrapped.

Operations are server-side only unless OpenAPI marks them
`x-client-accessible: true`. A backend token requires both the server claim and
`Stream-Auth-Type: server`. End users can access only sessions belonging to their
customer, user ID and caller kind. A server may open a session on behalf of a
named authenticated or guest user; an anonymous claim cannot take it over.

End-user limits default to 200 model responses and 5,000,000 tokens per UTC day,
counted against both user and IP. Set `ROUTER_RATE_LIMIT_MESSAGES_PER_DAY` and
`ROUTER_RATE_LIMIT_TOKENS_PER_DAY` to change them; `0` disables a limit. Backend
callers are excluded. Counters require Redis and fail open if it is unavailable;
proxy deployments enforce their limits at the gateway.

### Storage and multiple nodes

Postgres records provider requests, per-turn latency, cost rollups, agent configs
and configuration audit history. It does not hold the live agent process.
Stream Chat holds conversation messages. Custom voice audio lives in object
storage. `MEM0_API_KEY` enables long-term memory; `TURBOPUFFER_API_KEY` enables
knowledge retrieval.

Sessions live on one node. With Redis configured, event sockets relay between
nodes and session operations forward by gRPC over the API port. Set
`ROUTER_NODE_ADVERTISE` if peers cannot reach the automatically chosen address.
A dead owner produces 502 until its claim expires; live conversations are not
migrated to another process. Audio sockets stay on the node they connected to.

Stream resources use the deployment's app by default. For each customer's own
Stream app, see [Stream tenancy](docs/stream-tenancy.md).

To move database data, `cmd/router replicate --from <url> --api-key <key>
--api-secret <secret>` copies an app and follows its changes; `--once` copies only.
The source API also exposes `/v1/data/export`, `/v1/data/import` and
`/v1/data/changes`. Credentials and OAuth tokens are excluded, and voice buckets
must be copied separately. Change cursors expire after
`ROUTER_DATA_MOVE_RETENTION` (default seven days); an expired cursor returns 410
and requires a new export.

## Sessions and history

| Operation | Endpoint |
| --- | --- |
| Start a voice or text session | `POST /v1/agents/sessions` |
| Read or end a session | `GET` / `DELETE /v1/agents/sessions/{id}` |
| Send a prompt | `POST /v1/agents/sessions/{id}/respond` |
| Speak supplied text | `POST /v1/agents/sessions/{id}/say` |
| Interrupt speech | `POST /v1/agents/sessions/{id}/interrupt` |
| Observe events and handle client tools | `GET /v1/agents/sessions/{id}/events` (WebSocket) |
| Carry voice audio without a Stream call | `GET /v1/agents/socket` (WebSocket) |

A text session sets `"text": true` and omits `call_id`; it uses the same
instructions, tools and knowledge as a voice session without STT or TTS.
Use `config_id` to load a saved agent. An agent directory can contain
`agent.yaml`, `instructions.md`, skills, knowledge and simulations; the
[Go SDK's directory sync](../sdks/go/README.md#an-agent-as-a-directory) shows how
to upload it.

The audio socket starts with
`{"type":"start","sample_rate":16000,"session":{...}}`. After the session
response, binary frames carry mono PCM16 audio in both directions. A `cleared`
event discards queued playback. Closing the socket or sending `stop` ends the
session. Tool calls and other events use the session's events socket.

### Persistent conversations

With Stream credentials, text sessions persist to an `agent:support-<uuid>` Chat
channel unless `incognito` is set. Pass `conversation_id` to resume one. The
backend checks its customer and agent ownership and restores at most 100
messages / 60,000 characters of ordinary conversation. `context_truncated`
reports when history exceeded the limit. Stored tool attachments are never
executed or restored as instructions.

Read history through
`GET /v1/agents/conversations/{cid}/messages?agent_id=...&before=...`.
`conversation_updated` events carry cumulative text, tool status, timestamps and
`saved` / `persistence_error`. Intermediate updates are throttled to 200 ms;
initial messages, tool transitions and final replies are saved durably. Pending
writes retry in memory and can be lost if the process stops. Closing the last
client ends the session but leaves the saved channel available. Without Stream
credentials, new text sessions run without a channel.

Voice sessions also write settled caller transcripts and replies to Chat.
`ROUTER_CHAT_TIMINGS=true` or `cmd/agent -chat-timings` adds latency details to
reply messages; these annotations are excluded when history is read back.

### Other APIs

- **Individual modalities:** STT, TTS, LLM and search can be used without an
  agent. See the [SDK examples](../sdks/go/README.md#one-modality-at-a-time).
- **Short recordings:** STT/TTS `inline: true` completes synchronously without a
  database. The 202 response contains the result. Limits are 8 MiB audio,
  16,000 text characters and 90 seconds; URLs and callbacks are not accepted.
- **Phone calls:** `GET /v1/phone/vendors` reports supported operations. Inbound
  attachment, outbound dial and search filters differ by vendor. Use the
  [SDK phone example](../sdks/go/README.md#on-the-phone) and
  [vendor configuration](internal/phone/phone.yaml). Purchasing numbers incurs
  vendor charges. Existing SIP trunks can be registered with `/v1/phone/trunks`.
- **Audit history:** `POST /v1/audit/query` shows configuration changes. Server
  headers `X-Stream-Client`, `X-Stream-Actor-Id` and `X-Stream-Actor-Name` label
  those changes; they grant no permissions.

## Debug and develop

For a slow reply, inspect `GET /v1/agents/calls/{id}/timeline`. It separates
transcript settling, turn decisions, model calls, synthesis and first audio.
`reply_hold_ms` is already included in the end-to-end latency: do not add it
again. Missing stages are absent rather than zero. Model calls overlap other
stages and their durations should not be summed into the critical path.

`first_audible_frame_ms` measures the outgoing track consuming a non-silent
frame, not sound reaching the browser. `speech_end_to_audio_ms` is an estimate
using STT timing and excludes network transit and browser playback.
`GET /v1/turns/stats` aggregates turn latency; modality statistics measure
individual provider calls. Add `tags` such as `project=support` to break down
usage and cost.

| Symptom | Check |
| --- | --- |
| No eligible model | The target exists, its capabilities match and a candidate has credentials |
| Agent does not join | Stream credentials, call ID/type and router logs |
| EOT startup failure | Endpoint path, private-service credentials and permanent HTTP errors |
| Replies start too late | Timeline stages, EOT retries, model TTFT and first-audio hold |
| 401 / 403 from the API | Auth mode, token claims and whether the operation permits client access |
| Unsaved conversation | Stream Chat configuration and `persistence_error` events |

From `acceleration/`:

```bash
go test ./...                       # unit tests, no external services needed
go vet ./...
go test -tags integration ./...     # real providers, Postgres and Redis
```

Integration suites skip when their required configuration is absent. They may
send audio and prompts to configured providers. The conversation-quality suite
is `go test -tags integration -run TestConversationSuite ./internal/agent`.

The Go HTTP structs are the API source of truth. After changing an operation:

```bash
go run ./cmd/openapi
(cd ../sdks/go && go generate ./...)
```

Follow the [repository SDK policy](../AGENTS.md#sdk-changes) for other clients.
Do not edit `api/openapi.yaml` or generated SDK types by hand.

| Directory | Responsibility |
| --- | --- |
| `cmd/router`, `cmd/agent`, `cmd/chat`, `cmd/say` | Executables |
| `internal/api`, `internal/session` | HTTP/WebSocket API and session lifecycle |
| `internal/agent`, `internal/harness` | Audio cadence, turn decisions, replies and tools |
| `internal/*router`, `internal/routing` | Provider selection, failover and model definitions |
| `internal/config`, `internal/auth`, `internal/policy` | Deployment settings, identity and policy |
| `internal/store`, `internal/live`, `internal/node` | Persistence, Redis state and node forwarding |
