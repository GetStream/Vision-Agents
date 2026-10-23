# Model Router

A Go service that fronts several model providers, picks one per request, and records what
each customer used. Speech-to-text, text-to-speech and large language models are routed
today, all three over one modality-agnostic core.

On top of the three there is an agent: it joins a Stream call, transcribes what it hears,
answers with a model and speaks the reply back. Because it is built from the routers rather
than from provider instances, every turn of a conversation gets the same failover, health
and billing as a direct API call.

## Layout

| Path                 | What it is                                                        |
| -------------------- | ----------------------------------------------------------------- |
| `internal/routing`   | The modality-agnostic core: config, registry, selection, failover, stats |
| `internal/audio`     | `PcmData`, the PCM type every provider exchanges                   |
| `internal/emit`      | `Emitter[E]`, the buffered event channel providers hold            |
| `internal/stt`       | The speech-to-text contract: PCM in, transcript and turn events out |
| `internal/sttrouter` | Speech-to-text providers and sessions                              |
| `internal/tts`       | The text-to-speech contract: text in, PCM and synthesis events out |
| `internal/ttsrouter` | Text-to-speech providers and sessions                              |
| `internal/llm`       | The LLM contract: messages in, streamed text and token counts out  |
| `internal/llmrouter` | LLM providers and sessions                                         |
| `internal/sts`       | The speech-to-speech contract: the caller's audio in, the model's voice, transcripts and tool calls out |
| `internal/stsrouter` | Speech-to-speech providers and sessions                            |
| `internal/search`    | The search contract: a question in, an answer and its sources out  |
| `internal/searchrouter` | Search providers and sessions                                   |
| `internal/agent`     | The conversation: transcribe, answer, speak, with barge-in          |
| `internal/harness`   | Sits between the caller and the model: skills, delegation, cancellation |
| `internal/agent/streamedge` | The agent's transport: a Stream call over WebRTC             |
| `internal/chatlog`   | Writes what was said into a Stream Chat channel                     |
| `internal/memory`    | What an agent remembers between calls; `mem0/` is the one provider  |
| `internal/knowledge` | What the business wrote down; `turbopuffer/` is the one provider    |
| `internal/knowledge/ingest` | Cutting documents into passages, for both the command and the endpoint |
| `internal/knowledge/urls` | Keeping a knowledge base filled from pages published elsewhere   |
| `internal/phone`     | Telephony: the vendor contract, the Stream SIP trunk, the service   |
| `internal/phone/twilio`, `internal/phone/telnyx` | The two vendors that also answer on a number |
| `internal/phone/sinch`, `.../bandwidth`, `.../vonage`, `.../bird`, `.../plivo` | Five more that buy numbers and call out on them |
| `internal/phone/didww`  | Sells numbers and cannot dial: it has no call control API at all    |
| `internal/store`     | Postgres via bun, plus the goose migrations in `migrations/`        |
| `internal/live`      | Redis via rueidis: provider health and live per-customer counters   |
| `internal/api`       | HTTP layer: Huma operations on chi, rendered into `api/openapi.yaml` |
| `cmd/router`         | Serves the HTTP API                                                 |
| `cmd/say`            | Types a line, hears it                                              |
| `cmd/chat`           | Types a line, reads the answer                                      |
| `cmd/agent`          | Joins a Stream call and holds a conversation                        |
| `cmd/phone`          | Buys numbers, points them at an agent, calls out and transfers live calls |
| `cmd/phonecheck`     | Places a live test call and logs where each hop of it ran |
| `cmd/knowledge`      | Reads documents into a knowledge base an agent can look things up in |
| `deploy/parakeet`    | The streaming Parakeet Truss deployed to Baseten                    |
| `deploy/s2-pro`      | The streaming S2 Pro Truss, written and validated but not yet pushed |
| `deploy/breeze-tts-2` | The streaming Breeze TTS 2 Truss, written but not yet pushed       |
| `deploy/gemma-4`     | The Gemma 4 26B-A4B vLLM Truss, thinking off, live on Baseten `qvmmvrrq` |

Three providers are therefore unreachable until someone deploys them. `s2pro` is under the
Fish Audio Research License and wants an H100, so both questions are worth settling before
the push; the hosted `fish` provider serves the same model in the meantime. `breeze` is in
the same position under the BreezeBlue Research and Non-Commercial License, and there its
hosted API is the licensed alternative. `gemma` needs a dedicated deployment because Gemma
is not on Baseten's shared Model APIs. With `S2PRO_WS_URL`, `BREEZE_WS_URL` or
`GEMMA_BASE_URL` unset each fails to build and routing moves to the next candidate, so a
shortcut still resolves.

All three LLM providers speak OpenAI-compatible chat completions, so they share one
implementation in `internal/llm/openaicompat` and differ only in base URL, credentials and
extra request fields. `deepseek` reaches Baseten's shared Model APIs, which need no
deployment. The speech-to-speech providers split the same way: OpenAI, xAI and Alibaba
send OpenAI's realtime events and share `internal/sts/openairealtime`, differing in how
the session frame is spelled; Gemini Live is its own package under `internal/sts/gemini`.

## Build prerequisites

Everything builds without cgo. `cmd/say` needs `ffplay` at runtime (part of ffmpeg)
to play audio, or `-out` to write a file instead.

## Configuration

Everything that is not a provider credential is one YAML file, read by `internal/config`:

```bash
go run ./cmd/router --config /etc/router.yaml
```

Its keys are `addr`, `public_url`, `log_level`, `dashboard_url`,
`trusted_proxies`, `routing_config`, `phone_config`, `prices_file`, `voices_bucket_url`, and the nested
`postgres.dsn`, `redis.{addr,username,password}`, `node.{advertise,listen}`, `listen_fd`, `shutdown.drain`, `auth.{mode,kek}`,
`rate_limit.{messages_per_day,tokens_per_day}`, `data_move.retention`, `connectors.enabled` and
`stream.{api_key,api_secret}`. Naming no file loads one of the three embedded in the
binary, by `ROUTER_ENV`.

Every variable below still wins over the file, so a chart, compose and `.env` keep working
with nothing changed, and the effective settings are written back into the environment for
the other commands that read them there.

| Variable                | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| `ROUTER_CONFIG_FILE`    | Path to the deployment's own YAML, which is what `--config` sets |
| `ROUTER_ENV`            | `local` (default), `staging` or `testing`. Loads `internal/config/<env>.yaml` when no file of your own is named. The integration suites always use `testing`, which has its own `model_router_test` database and is the one file that wins over the environment |
| `ROUTER_ADDR`           | HTTP listen address, defaults to `:8080`                   |
| `ROUTER_POSTGRES_DSN`   | Postgres DSN. Without it, nothing is recorded              |
| `ROUTER_REDIS_ADDR`     | Redis `host:port`. Without it, routing ignores health, and a session is only reachable on the node running it |
| `ROUTER_NODE_ADVERTISE` | `host:port` this node's peers reach it at. Unset works it out from this host's own address and `ROUTER_ADDR`'s port, which is right wherever a pod is reachable from its peers. See [More than one node](#more-than-one-node) |
| `ROUTER_NODE_LISTEN`    | A second listen address, for this node's peers only. Unset means they use `ROUTER_ADDR`. Set it when two routers share one `ROUTER_ADDR`, so a request for a draining router's session still reaches it |
| `ROUTER_LISTEN_FD`      | A listening socket inherited from a supervisor, served instead of `ROUTER_ADDR`, so an old and a new router take connections from one queue |
| `ROUTER_SHUTDOWN_DRAIN` | How long a stopping router keeps its sessions after it stops taking work, e.g. `1h`. Unset ends them once HTTP has drained (10s). A second signal ends the wait. A draining router stops serving metrics when the drain starts, so a router started after it on the same `ROUTER_METRICS_ADDR` serves them |
| `ROUTER_VOICES_BUCKET_URL` | Bucket for voice recordings, e.g. `s3://voices?region=eu-west-1` or `gs://voices`. Without it, voices of your own are unavailable |
| `ROUTER_CONFIG`         | Path to a capability config; defaults to the built-in one  |
| `ROUTER_PHONE_CONFIG`   | Path to a vendor list; defaults to the built-in one        |
| `ROUTER_PRICES_FILE`    | Path to a price file of the deployment's own (`router prices check --file` checks one), laid over the built-in `prices/vendor.json`: the router prices from both, the file's entry winning for a model, and loads both into its database. Only the first load reads its `markup_bps`, and nothing applies a markup yet. Unset prices from and loads `vendor.json` alone. See [Prices](#prices) |
| `ROUTER_AUTH_MODE`      | `api_key` (default), `proxy`, `noauth` or `custom`. See [Authentication](#authentication) |
| `ROUTER_AUTH_KEK`       | Unseals the stored key secrets. Required by `api_key`, and held outside the database on purpose. Also version 1 of the connector keyring, and what a node's peers prove they are this deployment's with, so a deployment of more than one node needs it |
| `ROUTER_AUTH_KEK_V<n>`  | Version `n` of the keyring that seals connector credentials, e.g. `ROUTER_AUTH_KEK_V1`. Keep an old version until every row sealed under it is rewrapped |
| `ROUTER_AUTH_KEK_VERSION` | Which keyring version seals new connector credentials, defaults to `1`. Every `ROUTER_AUTH_KEK_V<n>` that is set still opens its rows, so moving this back to an older version is safe |
| `ROUTER_CONNECTORS_ENABLED` | `true` turns connectors on. The router then needs the keyring version `ROUTER_AUTH_KEK_VERSION` names, in every auth mode, and refuses to start without it. Off by default |
| `ROUTER_CONNECTORS_PROXY_CALLS_PER_MINUTE` | Direct calls one app may send to one connector through the connection proxy in a minute, counted in Redis across routers, defaults to `60`. A call over it is a 429 with `Retry-After`. `0` turns it off, and so does a router with no Redis |
| `ROUTER_RUNS_CONCURRENT_PER_APP` | Headless runs (`POST /v1/agents/runs`) one app may have in flight, counted in Redis across routers, defaults to `20`. A run over it is a 429 `concurrency_limited` with `Retry-After`. `0` turns it off, and so does a router with no Redis |
| `ROUTER_CODE_SANDBOX_BROKER` | What holds a connector's credential for code in a sandbox: `daytona` (Daytona Secrets), the default, or `proxy`, the egress proxy (`internal/sandbox/egress`), which needs Redis, `ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL`, `ROUTER_CODE_SANDBOX_PROXY_CA_CERT`, and the proxy served by the same router (`ROUTER_CODE_SANDBOX_PROXY_ADDR`, `ROUTER_CODE_SANDBOX_PROXY_CA_KEY`). Behind the proxy a sandbox reaches the proxy alone, and the proxy reaches its grants' hosts and the config's `allowed_domains`; a config with `network: true` and no `allowed_domains`, or with a `network_allow_list`, is refused when a session with a sandbox binding starts |
| `ROUTER_CODE_SANDBOX_SECRET_PREFIX` | Names this router's environment in every Daytona secret it makes, such as `stg` on staging (lowercase letters and digits only), since routers sharing a Daytona organization must never touch each other's. A session with a `sandbox: true` binding is refused without it |
| `ROUTER_CODE_SANDBOX_PROXY_ADDR` | Where the router serves the sandbox egress proxy (`internal/sandbox/egress`), such as `:8091` on staging. Unset, the default, serves none. Needs Redis and connectors on |
| `ROUTER_CODE_SANDBOX_PROXY_CA_CERT`, `ROUTER_CODE_SANDBOX_PROXY_CA_KEY` | The PEM certificate and private key of the CA the proxy signs each granted host's certificate with. The router refuses to start with `ROUTER_CODE_SANDBOX_PROXY_ADDR` set and either missing, or a certificate setting that holds a private key. Only the certificate goes into a sandbox |
| `ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL` | The proxy as a sandbox reaches it, `http://<public IPv4 address>:port`, for `ROUTER_CODE_SANDBOX_BROKER=proxy`. The router refuses to start with that broker and this, the CA certificate, the proxy address or the CA key missing, or with a host name or an IPv6 address here: the sandbox reaches that address alone and has no DNS |
| `<ENV>_MCP_CLIENT_ID`, `<ENV>_MCP_CLIENT_SECRET` | This deployment's own OAuth client for a connector whose manifest has `client.env: <ENV>` and lists `operator` in `client.registration`, such as `SLACK_MCP_CLIENT_ID` for `slack`. Read at each consent and refresh. An app's own client is put through `PUT /v1/agents/connectors/{id}/oauth-client` instead |
| `ROUTER_LIMITS_MODE` | `enforce` (default), `shadow` or `off`: how much of credit, the spend hard limit and the customer's spend limits the router enforces. `shadow` works every refusal out, logs it as `would refuse` and admits the request; `off` refuses nothing for money, and work is still priced, recorded and shipped. Both log a warning at start. With limits on, the router refuses to start without `ROUTER_POSTGRES_DSN`, and in `proxy` mode without `ROUTER_USAGE_URL`. See [Admission](#admission) |
| `ROUTER_USAGE_REDIS_URL` | The Redis the usage service mirrors every subject's spend to (its `USAGE_REDIS_URL`), which the customer's spend limits are read from every second. Needs `ROUTER_USAGE_URL`. Unset, they are read from the usage service itself every 5 s. See [Spend limits](#spend-limits) |
| `ROUTER_RATE_LIMIT_MESSAGES_PER_DAY` | Model responses one end user may ask for in a UTC day, defaults to `200`. `0` turns it off. It is the default: an app's policy sets its own (`quota`). See [Daily limits](#daily-limits) |
| `ROUTER_RATE_LIMIT_TOKENS_PER_DAY` | Tokens one end user may spend in a UTC day, defaults to `5000000`. `0` turns it off. It is the default: an app's policy sets its own (`quota`) |
| `ROUTER_SPECULATIVE_REPLIES` | `true` starts each reply while the flow controller is still deciding whether the words were meant for the agent, and holds it until the ruling says to answer. Saves the ruling's round trip on answered turns and pays for the replies a ruling drops. Off by default |
| `ROUTER_CHAT_TIMINGS` | For development: each voice reply in the session's chat channel gets a line after its text with how long its turn took, and the same figures as a `timings` field on the message. See [Transcripts](#transcripts-memory-and-phone). Anything that reads the conversation back leaves the line out. Off by default; `cmd/agent` takes it as `-chat-timings` |
| `ROUTER_REPLY_HEDGE` | How long a reply may say nothing, neither text nor a tool call, before the same request is also asked of another candidate of its target, and whichever says something first is kept. The other candidate is on a different model when the target has one, and on a different provider when it has one, and is never one the router has found unavailable. The one that loses is cancelled the moment it does and still read out, so both calls are reported as model calls and recorded, the cancelled one as any cancelled call is. A reply is hedged once, and a hedge that fails leaves the first request going. Only replies and the flow controller's decisions (`ROUTER_FLOW_HEDGE`) are hedged, not the guardrails or background work, nor a reply that continues from a response the provider holds. A target with no other candidate is asked once. Defaults to `1200ms`; `0` turns it off. `cmd/agent` takes it as `-reply-hedge` |
| `ROUTER_FLOW_HEDGE` | `ROUTER_REPLY_HEDGE` for the flow controller's decisions, which the caller waits on in silence before the reply is asked for. A decision is a few tokens, so one that has said nothing by then has stalled rather than thought, and the same request is asked of another candidate of `llm-flow` as well; whichever says something first rules on the turn. Without it a stalled decision waits out the controller's 3 s deadline and the turn is answered unruled. Defaults to `400ms`; `0` turns it off. `cmd/agent` takes it as `-flow-hedge` |
| `ROUTER_MOCK_PROVIDERS` | `true` adds `mock/mock-stt`, `mock/mock-tts` and `mock/mock-llm`, which answer from nothing, reach no vendor and bill nothing, for load testing the router itself (see `performance_benchmark/`). No capability alias routes to one, so a session only gets a mock by naming it. Off by default |
| `ROUTER_TRUSTED_PROXIES` | CIDR ranges your own proxies sit in, comma separated, e.g. `10.0.0.0/8`. Decides how much of `X-Forwarded-For` is believed. Unset means none of it is, and the connection's address is used |
| `ROUTER_DATA_MOVE_RETENTION` | How long recorded changes are kept while a customer moves between deployments, defaults to `168h`. See [Moving a customer](#moving-a-customer) |
| `ROUTER_LOG_LEVEL`      | `debug`, `info` (default), `warn` or `error`               |
| `HARNESS_SKILLS`        | Path to a skill set; defaults to the built-in one          |
| `MEM0_API_KEY`          | mem0 credentials. Without it the agent remembers nothing   |
| `TURBOPUFFER_API_KEY`   | turbopuffer credentials. Without it an agent looks nothing up |
| `TAVILY_API_KEY`        | Tavily credentials, one of three ways to find out what is true today |
| `EXA_API_KEY`           | Exa credentials. Also what reads a URL into a knowledge base |
| `PERPLEXITY_API_KEY`    | Perplexity credentials, for the ranked index and for Sonar |
| `DAYTONA_API_KEY`       | Daytona credentials. Without it a session cannot ask for a sandbox |
| `TWILIO_ACCOUNT_SID`    | Twilio credentials, for buying and operating numbers       |
| `TWILIO_AUTH_TOKEN`     | Twilio credentials                                         |
| `TELNYX_API_KEY`        | Telnyx credentials                                         |
| `TELNYX_CONNECTION_ID`  | The Telnyx TeXML application numbers and calls are on. Calls to the webhook from any other application are hung up |
| `TELNYX_PUBLIC_KEY`     | Telnyx's public key (base64), to check the calls it sends to `/v1/phone/hooks/telnyx`. Without it Telnyx numbers are not answered. Telnyx numbers also need `ROUTER_AUTH_KEK`: without a keyring they cannot be attached |
| `DEEPGRAM_API_KEY`      | Deepgram Flux credentials                                  |
| `CARTESIA_API_KEY`      | Cartesia credentials, used by Sonic and the Ink 2 transcriber |
| `CARTESIA_VOICE_ID`     | Optional default voice; a built-in one is used when unset  |
| `ELEVENLABS_API_KEY`    | ElevenLabs credentials, used by the three voices and the Scribe v2 Realtime transcriber |
| `ELEVENLABS_VOICE_ID`   | Optional default voice; a built-in one is used when unset  |
| `INWORLD_API_KEY`       | Inworld credentials, used by TTS 2 Flash and the STT-1 transcriber |
| `FISH_API_KEY`          | Fish Audio credentials                                     |
| `FISH_VOICE_ID`         | Optional Fish reference id to clone a voice from           |
| `OPENAI_API_KEY`        | OpenAI credentials                                         |
| `GOOGLE_API_KEY`        | Gemini credentials, from AI Studio; used by the LLM and the transcriber |
| `BASETEN_API_KEY`       | Baseten credentials, for the Model APIs and both deployments |
| `XAI_API_KEY`           | xAI credentials, used by the Grok transcriber              |
| `TOGETHER_API_KEY`      | Together AI credentials, used by the Together-hosted Parakeet and both Nemotrons |
| `PARAKEET_WS_URL`       | The Parakeet WebSocket endpoint                            |
| `S2PRO_WS_URL`          | The S2 Pro WebSocket endpoint. Not yet deployed, see above |
| `BREEZE_WS_URL`         | The Breeze TTS 2 WebSocket endpoint. Not yet deployed, see above |
| `DEEPSEEK_BASE_URL`     | Optional; overrides Baseten's shared Model APIs endpoint    |
| `GEMMA_BASE_URL`        | The Gemma 4 deployment endpoint, `https://model-qvmmvrrq.api.baseten.co/environments/production/sync/v1` |
| `GEMMA_API_KEY`         | Optional; the bearer token for `GEMMA_BASE_URL` when it is not on Baseten. `BASETEN_API_KEY` is used when unset |
| `STREAM_API_KEY`        | Stream credentials, used by `cmd/agent` and to say which app a browser joins |
| `STREAM_API_SECRET`     | Stream credentials; the agent mints its own token from them, and the router mints a browser's |
| `STREAM_USER_TOKEN`     | Optional; used in preference to the secret                 |
| `EXAMPLE_BASE_URL`      | Optional; points `cmd/agent`'s demo link at another deployment |

## Authentication

`ROUTER_AUTH_MODE` picks between the four deployments this serves.

`api_key`, the default, verifies the caller itself. `proxy` believes what something in
front of it already decided. `noauth` asks for nothing and treats every caller as the
customer's own backend. `custom` is an authenticator the deployment supplies in code.

The default is the one that verifies something, because a default that trusts whatever
reaches it is only correct when something else guarantees nothing does, and that guarantee
is a NetworkPolicy in one deployment and nothing at all in the next. `api_key` refuses to
start without a store and a key encryption key, and says which is missing, rather than
starting and refusing every request for a reason only visible in a 401.

`api_key` needs Postgres to look keys up in. A request
carries the key in `X-Api-Key` and a token in `Authorization: Bearer`, signed HS256 with the
secret belonging to that key; a socket takes the same two as `api_key` and `token`. The
proxy headers are ignored entirely in this mode, since reading them would be a way around
the key. Keys belong to apps and apps belong to organizations, and it is the app id that
every other table's `customer_id` holds.

A key is two values with different jobs. The id is public — `vak_live_…`, ending in a
checksum so a truncated paste is rejected before the database is asked, and it is what a
log line names so an operator can revoke the right one. The secret is `vas_live_…`, shown
once and then held only sealed: verifying a token means recomputing its signature, which
means the secret cannot be hashed, so it is encrypted with AES-256-GCM under
`ROUTER_AUTH_KEK` and a leaked backup yields ciphertext. `kek_version` is there so that key
can be rotated by re-wrapping rows rather than by reissuing every secret.

Every failure is one 401 with one body, because a caller that could tell an unknown key
from a bad signature could use the difference to find out which keys exist. Revoking a key
keeps the row: a year of request rows pointing at a deleted key is unattributable noise.

`proxy` names the caller by `X-Stream-App-Id` and `X-Stream-Organization-Id`, the end user
by `X-Stream-User-Id` and the sort of caller by `Stream-Auth-Type`. Nothing is verified, so
it is only safe where a proxy in front has already authenticated the caller and nothing
else can reach the router. That proxy owns rate limiting, and it has to overwrite all four
rather than pass a caller's own through.

`noauth` reads `X-Customer-Id`, or a `customer_id` query parameter on a socket since a
browser WebSocket carries no headers, and takes every caller for that customer's own
backend. It deliberately reads neither the user nor the auth type: nothing here verified
them, so believing them would only let a caller name itself anything it liked, and a user
id is what one person's conversations are kept from another's by. It is for a laptop, and
it says so in a warning at startup.

`custom` builds nothing. A deployment embedding this module passes its own
`auth.Authenticator` to `api.NewServer` through `api.WithAuthenticator`; `auth.Func` makes
one out of a function. The stock router binary has none compiled in and refuses to start in
this mode rather than pretending.

### Server-side only

Everything is server-side only unless the spec says otherwise. An operation a device may
reach is marked `x-client-accessible: true`, and six are:

| Operation | Why a device may |
| --- | --- |
| `GET /v1/search` | A question and its answer are one round trip, for whoever asked |
| `POST /v1/agents/sessions` | Opening a conversation is the point |
| `GET /v1/agents/sessions` | Finding the conversation it already has |
| `GET /v1/agents/sessions/{id}` | Reading the one conversation it is having |
| `DELETE /v1/agents/sessions/{id}` | Hanging up on its own |
| `GET /v1/agents/sessions/{id}/events` | The conversation itself, as it happens |

The default is inverted rather than enumerated because the failure modes are not
symmetrical: a new operation nobody marked is refused to devices until somebody decides it
should not be, and the cost of getting that wrong is a 403 in front of a feature. Listing
what to close instead means a new operation is open by accident, and the cost of getting
that wrong is a config a phone can rewrite. The generated SDKs and the check in front of the
handlers both read the same marks, so there is one answer rather than two that can drift.

Unauthenticated paths — health, the plugin OAuth callback — declare `security: []` and sit
outside this entirely: there is no caller to classify.

A server-side caller says so twice. `Stream-Auth-Type: server` is the declaration, and a
token carrying `server: true` and no `user_id` claim is the proof; `jwt` and a `user_id` are
what a token minted for an end user carries, and are what a caller that names neither is
taken to be. Both are required because each fails closed in a different direction: nothing
signs the header, so it cannot promote a request on its own, and a server token pasted into
a browser that sends the client header is used by a browser and treated as one. Stream's own
convention is the same one, so a token minted by a Stream server SDK already fits.

The auth type has no query parameter, unlike the key and the token. A browser WebSocket
cannot set a header, so there is nothing for one to say, which is what stops a browser
opening a dispatch socket and answering somebody else's callers.

The refusal is a 403 rather than a 401: the caller has already proved who it is, so a
specific answer gives nothing away, and one told "unauthenticated" would go looking for a
credential problem it does not have. In `proxy` the auth type is believed the same way the
app id is, and a caller that names none is taken to be a backend, since a proxy that has
classified a caller says so. The gateway overwrites the header from the credential it
verified, so a caller cannot name its own. In `noauth` there is nothing to declare: every
caller is a backend because nothing there could make it anything else.

A backend may still say which of its users it is working for, in `X-Stream-User-Id`. A
server token names no user by definition, so the header is the only place to put it, and it
is unsigned because a caller holding the secret could mint a token for that user anyway.
What it buys is that the session the backend opens belongs to that user, so the user's own
device can reach it afterwards. It costs nothing: the caller is still server-side, still
reaches everything, and is still charged no daily limit.

Beside it, a backend may say the email that user signs in with, in `X-Stream-User-Email`. The
router keeps the latest one per user and reads it from a server-side caller only, since a
device that could set it could claim another person's address. A connector whose manifest sets
`owner_email` (nessy) refuses a user-owned connection's first consent, storing nothing, unless
the provider's `id_token` names that email (AI-1543).

### Who a session belongs to

Sessions are the one thing a device opens, so they are the one thing that needs a boundary
inside a customer. A session records who opened it — the customer, the user id and which
sort of caller claimed it — and every later request for it is matched against that.

A caller is one of four kinds, decided by the credential and not by the request:

| Kind | How | Reaches |
| --- | --- | --- |
| server | `Stream-Auth-Type: server` and `server: true` | every session of its customer |
| authenticated | a token naming a `user_id` | its own user's sessions |
| guest | a token naming a `user_id`, minted for a guest | its own guest's sessions |
| anonymous | no token, a user id merely claimed | the sessions opened under that claim |

The three that are not `server` are levels of end user, and an app may turn the bottom two
away. That is a `settings` document on the app row — `allow_anonymous` and `allow_guest`,
both defaulting to yes — read in the same query that resolves the API key. A level an app
refuses is answered 403 at the door rather than given a narrower API, so a new endpoint
cannot forget, and a missing key or a missing app row admits: `proxy` and `noauth` resolve
no app row here at all, so the levels are enforced in `api_key` mode only.

The kind is part of the identity and not a property of it, which is what makes an anonymous
claim harmless: an anonymous caller saying `user_id=alice` is stored as anonymous alice and
never matches the alice a token proved, so claiming a name buys nothing. That is also why
guest and authenticated are separate kinds rather than one "has a token" — a guest id is
issued per device and reused by nobody, but nothing in a token distinguishes the two, so the
declaration does.

An anonymous caller naming no user at all is refused a listing rather than shown a customer's
sessions: it has opened nothing that can be found again, and a caller that names nothing is
not the same caller twice.

Reaching a session means the customer matches and either the caller is server-side or both
the user id and the kind match. The one exception is a session a backend opened in a named
user's name: an authenticated or guest caller of that name reaches it, because a backend
opening the conversation and handing over the id is the ordinary shape of an integration.
Anonymous is left out of that, since an anonymous name is a claim nobody checked and
allowing it would make guessing whose a session was enough to read it.

`internal/session` enforces all of this rather than the handlers, so `getSession`,
`stopSession`, `deleteSession`, the events socket, the call token and the session actions cannot each be
wrong in their own way.

Not built yet, in the order [.factory/features/auth.md](../.factory/features/auth.md) puts
them: an endpoint that creates a key or writes a setting, scopes, and the single-use ticket
that would get the credential out of a socket's query string.

### Running behind a proxy

`proxy` exists for one deployment in particular. On Stream's own infrastructure
the router runs with authentication off, reachable only from `stream-accelerate`,
a proxy in the chat repository that verifies a Stream API key and the token
signed with that app's secret, rate limits the caller, and sets the two headers
above. It authenticates against Stream's own app records, so a customer holds
one credential rather than a second one issued by this service.

That proxy is not part of this repository, and does not need to be: authenticating
a Stream key means reading Stream's tables. Running this service on your own
means `api_key` mode instead, where the router authenticates callers itself.

### Daily limits

A token is spent by whoever asks for it, so an end user's device is capped and a
process the customer runs is not. A backend was given its own key and is trusted
with the spend that goes with it; a browser holding a token that backend minted is
not, because a leaked one is a stranger with your model bill.

Two things are counted, per UTC day, and reaching either refuses the next response:

| Counted | Default | Why |
| ------- | ------- | --- |
| Model responses | 200 | What "200 messages a day" means |
| Tokens | 5,000,000 | A backstop for the caller who asks for few responses and makes each one enormous. A turn of an agent with a few MCP servers carries tens of thousands of tokens of tool definitions, so it should not be what an ordinary day hits first |

These numbers are the deployment's default. An app sets its own as `quota` in its policy,
`PUT /v1/policies/app` (`messages_per_user_per_day`, `tokens_per_user_per_day`), above the
default or below it, and `0` is no cap. The PUT replaces the whole policy, so one that leaves
`quota` out clears it and the default applies again. Organizations have none: a `quota` sent
to `PUT /v1/policies/organization` is a 400. Each number is decided on its own. A limit Stream
sets on a whole app is the gateway's (AI-1253), a staff spend cap is AI-1254, and a history of
policy changes is AI-1255.

A turn is counted on the day of whoever sent it. In a session a backend opened in a user's
name, the backend's own turns are not counted, and what the user sends into it with their own
token is counted on their day. In a session the user opened, every turn is counted on their day.

Each is counted against two buckets: the `user_id` the caller's token names, scoped
to the customer who minted it, and the address the request came from, scoped to the
customer too, since each has its own limit. Both matter.
A user id is the honest answer to who is asking, and the address is the answer that
still holds when a backend mints a fresh user id per request.

What a backend asks for is never counted, even when it names the user it acts for, and
neither are the backend's own turns in a session it opened. Answers from `mock/*` models are not
counted either. A counted HTTP answer carries `X-Quota-Limit`, `X-Quota-Remaining`
and `X-Quota-Reset` (the unix time of the next UTC midnight), for the limit nearest to
refusing the caller.

The limit is enforced where a response is created, not only where a request is
admitted. A socket goes on sending frames and a call goes on talking long after
whatever opened them was let through, so an admission check alone would cap the
first turn of an unbounded conversation. A caller with nothing left gets a 429 with
a `Retry-After`, or an `error` frame if the socket was already open.

Counting happens in Redis, so a deployment without it caps nothing, and a Redis
that cannot be reached allows the request rather than refusing it. That is the
right way round: refusing everybody because a counter is missing is a worse outage
than the one the limit prevents.

`ROUTER_TRUSTED_PROXIES` is what makes the address half meaningful. `X-Forwarded-For`
is a list each hop appends to, and the leftmost entry is whatever the caller claimed,
so it is walked from the right and only entries written by proxies you named are
stepped past. Unset, the header is ignored entirely — correct with nothing in front,
and wrong behind a load balancer, where every caller would look like the balancer
and share one allowance.

### Organization and app policies

`GET`/`PUT /v1/policies/app` and `/v1/policies/organization` hold these settings, all
optional. The organization's are a floor its apps can tighten but not loosen.

| Setting | What it does |
| ------- | ------------ |
| `data_policy` | `allow_training` and `retention`, applied to every routed request as a floor under whatever it asked for. The stricter of the two scopes wins. A model that declares no `data_policy` meets no floor, so this narrows LLM, search and image routing to declared models |
| `quota` | On an app's policy only: what one end user may spend in a UTC day, in place of the deployment's default. An organization's policy that sets it is a 400. See [Daily limits](#daily-limits) |
| `prompt_injection` | Screens the newest input of every LLM response (the user's turn and any tool results) with the decision model router, using the default route (Jev), at the same time as the model call. Deltas are not held; the end of the response waits for the verdict (at most 2s), and an injection fails the response before its tool calls can be acted on |

The screen asks one question per harm in OpenRouter's prompt injection list (instruction
override, privileged modes, system override, prompt extraction, role manipulation, DAN-style
jailbreaks, safety bypass, role spoofing, control tokens), and decodes base64, hex and
letter-spaced text first so an encoded attack is judged on what it says.

An organization's policy applies to the apps the router has seen that organization name. Behind
a proxy that is the `X-Stream-Organization-Id` header; with API keys it is the key's app.
`noauth` reads no organization, so only app policies apply there. Every read of a policy fails
open. A policy has no `budget` any more: one sent is a 400 `budget_moved`, and spend limits are
their own table (below).

### Spend limits

The customer's spend limits are rows of the `limits` table, one per subject, level and interval
(`daily`, `weekly` from Monday, `biweekly` from Monday 2024-01-01, `monthly`, all UTC). A
subject is what spends: the organization, an app, an API key, an end user of an app, a usage
category across the organization, or one within an app (a team is the category
`team:<name>`). A row at a level above its subject is a default for each subject of its kind
there: an app-level `api_key` row limits every key of the app on its own, a row naming the key
wins over it, and the app's wins over the organization's. Work is refused once any subject it
counts for has spent its limit in the current window, so the organization's limit still refuses
an app whose own has room: 402 `spend_limit_reached`, or 429 `user_limit_reached` with
`Retry-After` for a limit on one end user, which stops that user alone. A reset
(`usage_resets`) counts a subject's spend from what it had spent when it was made, until the
window ends; the usage records are untouched. An origin (`hook:chat` and the rest) is no key a
limit counts, and a user id the usage records leave off is no user.

They are written through `/v1/spend/limits`, server-side only: one server-side credential of
any app manages every limit of its organization (an app seen in no organization is its own).

| Operation | Path | What it does |
| --------- | ---- | ------------ |
| `listLimits`, `createLimit` | `GET`, `POST /v1/spend/limits` | The organization's limits by id, filtered by subject, level, ids, team, category or interval; a new one, 201 |
| `getLimit`, `updateLimit`, `deleteLimit` | `GET`, `PATCH`, `DELETE /v1/spend/limits/{limit_id}` | One limit; its amount and note only; 204 |
| `getLimitUsage` | `GET /v1/spend/limits/usage` | One subject's current window in every interval: spent, the limit in force, its baseline and room, and the operator's spend hard limit beside them |
| `resetLimitUsage` | `POST /v1/spend/limits/usage/reset` | Resets the subject's spend for its limits in the windows named, with a reason |

A team is `subject: team` (or `app_team` with an `app_id`) and `team: finance`, kept as the
category `team:finance`; a category starting `team:` is a 400 `use_team`. A level left out is
the most specific the ids name. A limit above the one, in the same interval, of a subject that
contains it is a 400 `limit_above_parent` naming it (the organization contains its apps and
categories, an app its keys, users and categories, and an organization's category the same
category in an app); lowering a parent is allowed, and every read shows
`effective_limit_micros`, the lowest of the row's and its parents'. A second limit on the same
subject, level and interval is a 409 `limit_exists` naming the first. Writes are audited as
`limit`. The router that wrote a limit holds to it at once, the others within 5 s. Staff reach
the same operations for support under `/v1/manage/spend/limits`, with the manage key and an
`organization_id`; what they write is the customer's.

The organization and app budgets that policies held moved into this table with the migration
`20261112120000_limits`: each became the organization's or the app's own limit on its interval,
an app's under the organization it was last seen in, or under itself when it was seen in none.
The migration stops while an hourly budget exists, since limits have no hourly interval. Coming
down puts one budget back into each policy from its scope's own limits.

### Admission

Before every routed session and every LLM response the router answers from memory what the
customer may do, so an agent's speech never waits on a database:

- An app's policies are read through the configuration cache the first time it is seen, and
  again in the background every 5 s while it is in use. The authenticated request that first
  names an app starts that read, so its first session usually finds it done.
- Its organization's money is read every 5 s, for every organization in use at once, from the
  usage service (`ROUTER_USAGE_URL`), 100 organizations a call: the status, the balance, the
  spend hard limit and this day's and month's spend. The customer's spend limits are read on a
  loop of their own, so a slow Postgres never makes a balance stale: the organization's limits
  and resets every 5 s, and what each subject with a limit spent from the usage service's
  counters every second (`ROUTER_USAGE_REDIS_URL`, the Redis the service mirrors its windows
  to), or from the service itself every 5 s when that is not set, when a counter is missing,
  and for 5 s after that Redis fails a read.
  With no usage service there is no credit or spend hard limit, and each limited subject's
  spend is summed from the request rows every 10 s, four at a time.
- What this router recorded since is held by usage id from the moment the row is recorded,
  and each ledger lets go of it on its own read: against the balance and the spend hard
  limit, once a standing asked for after the usage service had the row lands (its ack, or the
  shard showing the row shipped, as by another router's sweeper); against the spend limits of
  every subject the row counts for, once a read of the counters asked for after an ack that
  says the counters took the row lands (or a read of the service itself after any ack), or
  with no usage service once a sum asked for after the shard wrote the row lands. Counted
  twice for a moment, never not at all. A row a ledger is never known to have is let go 10
  minutes after the later of when it was held and when that ledger last failed to read, and
  logged; while no read lands, every hold is kept. While the usage service acks none of an
  organization's rows, its limits' holds are kept too, and their 10 minutes start once acks
  come back. A limit counts only what was held in its current UTC window; the balance counts
  everything held.

In that order: a suspended organization is refused everything, 403 `account_suspended`. A
balance at or below what is held refuses work that costs money, 402 `insufficient_credits`.
The organization's spend hard limit, per UTC day and calendar month, refuses it however much
credit is left, 402 `system_limit_reached`; nothing a customer writes lifts it. Then the
customer's spend limits, subject by subject (the organization, the app, each category in the
organization and then in the app, the key, the user), 402 `spend_limit_reached`, or 429
`user_limit_reached` for the user. A model the deployment runs itself, or one that is free, is
still served past every one of them but a suspension. An organization is admitted until its
first read lands. Once it has gone a minute without one, its tier's `fail_mode` decides: a
closed tier refuses work that costs money until the balance can be read, an open one is
admitted on the last read. The customer's limits are read before the first admission that
needs them: an organization's the first time this router sees it, and what a subject with a
limit spent the first time it sees the subject, waiting at most 250 ms. A read that does not
land by then, or limits and spend last read over a minute ago, are the tier's `fail_mode` to
answer as well (an organization with no tier, or no usage service, follows the default tier,
which fails closed). The metrics are `router_admission_refresh_errors_total`,
`router_admission_oldest_read_seconds` and, in `shadow`, `router_admission_would_refuse_total`
by reason.

Telephony asks the same admission: buying a number, placing or transferring a call, and every
text an agent sends or answers on a line. Whatever a phone vendor charges for is billed work,
so a refusal of money refuses it (402, or 403 `account_suspended`). A call to one of the
organization's numbers is refused: on Telnyx it is rejected before it is answered, so it is
not billed. Over Stream SIP the bridge has already answered the caller when the router hears
of the call, so no agent is dispatched and the call is ended through Stream at once; the few
seconds it was up are billed by the vendor. The same holds for a telephony suspension.
Releasing a number is never refused. A campaign is refused at start, and one refused midway
pauses with the refusal's code as `paused_reason`, giving the contact it was about to ring
back to be rung once it is started again. A scheduled task that could not be delivered for
money records the refusal's code.

Holds are bounded in memory: an account keeps at most 10,000 apart and folds the oldest,
remembering up to 100,000 folded usage ids (about 15 MB per account at most), and the process
remembers at most 500,000 folded ids across every account (about 60 MB). A folded id past
either cap is forgotten (`router_admission_folded_ids_forgotten_total`): it is then held
until its fold's timer rather than its ack, which only ever refuses sooner.

Provider capabilities, prices and the capability shortcuts live in
[internal/routing/router.yaml](internal/routing/router.yaml), one section per modality.
Adding a provider or model is a config edit.

## Targets

A request names either a concrete `provider/model` or one of four capability shortcuts,
which mean the same thing in every modality:

| Shortcut                     | What it asks for                                    |
| ---------------------------- | --------------------------------------------------- |
| `en-low-latency`             | English, fast enough for a live conversation         |
| `multilingual-low-latency`   | More than one language, still fast                  |
| `en-high-accuracy`           | English, quality over speed                         |
| `multilingual-high-accuracy` | More than one language, quality over speed          |

A shortcut normally describes what it wants and takes whoever meets it, so adding a model is
a config edit and nothing else. `en-low-latency` for speech-to-text is the exception: every
live transcriber is realtime, low-latency and speaks English, so nothing any of them declares
tells one from another, and it names both Deepgram Flux models, Grok and Muse outright with
`only`. Gemini, the two Parakeets, Ink 2, Inworld and Scribe stay reachable by
`provider/model` and through the other shortcuts.

### The default groups

These are the shortcuts offered as a choice, for the caller who wants the models we would
pick rather than a list of their own to keep:

| Group                | What it is                                                                      |
| -------------------- | ------------------------------------------------------------------------------- |
| `llm-conversational` | Super lightweight: Gemma 4 26B-A4B and GPT-5.6 Luna, pinned to Gemma 4 26B |
| `llm-fast`           | The low-latency tier, pinned to Gemini 3.8 Flash                                |
| `llm-smart`          | The high-quality tier, pinned to GPT-5.6 Sol                                    |
| `stt-fast`           | Flux, Ink 2, Inworld and Nemotron, pinned to `deepgram/flux-general-en`         |
| `stt-accurate`       | Muse, Scribe v2 Realtime, Ink 2 and Nemotron, pinned to Muse                    |
| `tts-fast`           | The low-latency tier, pinned to Cartesia Sonic 3.6                              |
| `tts-quality`        | The top of the Artificial Analysis arena: ElevenLabs v4, Sonic 3.6, Inworld TTS-2 Flash, Breeze TTS 2 and ElevenLabs v3 Conversational |

```yaml
stt:
  providers:
    - stt-accurate
```

The ones that name their members do so with `only`, which is the thing the field exists for and the thing it
warns about: a name is a judgement about which models are wanted rather than a fact about what
they can do, so it goes stale in a way a requirement does not. That is what these are — an
opinion with a date on it — and the date is in the config beside them.

A group is its models and nothing else. It carries no terms, so a config that wants the label
still writes `diarize: true`, and the accurate group then narrows to Muse, which is the one of
its four that declares it and the one the pin already starts with. Both groups are the live
path only; a recording belongs at `en-recorded`, where the batch models are.

Speech-to-text also keeps its sprint-1 names (`en-realtime-best` and friends) as synonyms.
LLM adds two of its own. `llm-fast` is a fast answer, in whatever language, and `llm-thinking`
is what the skills run on: the part of a turn the talking model could not answer itself, which
the conversation carries on without. Both name the model this deployment wants rather than
leaving the choice to the ranking — `gemini/gemini-3.8-flash` and `openai/gpt-6.1-sol` — and
only reach the rest of their tier when that model is unavailable or fails to start.

Which models those shortcuts reach for LLM, and what each is billed at per million tokens
(the vendor rates in `prices/vendor.json`):

| Model                            | Tier         | In     | Cached  | Out     |
| -------------------------------- | ------------ | ------ | ------- | ------- |
| `deepseek/DeepSeek-V4-Flash-0731` | low-latency  | $0.13  | $0.028  | $0.26   |
| `openai/gpt-5.6-luna`             | low-latency  | $0.20  | $0.02   | $1.20   |
| `openai/gpt-6-luna`               | low-latency  | $0.10  | $0.01   | $0.50   |
| `gemini/gemini-3.8-flash`         | low-latency  | $0.75  | $0.075  | $3.75   |
| `gemma/gemma-4-26B-A4B-it`        | low-latency  | $0.24  | -       | $1.20   |
| `deepseek/DeepSeek-V4-Pro-0813`   | high-quality | $1.32  | $0.132  | $3.96   |
| `openai/gpt-5.6-terra`            | high-quality | $2.00  | $0.20   | $12.00  |
| `openai/gpt-5.6-sol`              | high-quality | $5.00  | $0.50   | $30.00  |
| `openai/gpt-6-sol`                | high-quality | $2.00  | $0.20   | $10.00  |
| `openai/gpt-6.1-sol`              | high-quality | $2.00  | $0.10   | $10.00  |
| `openai/gpt-6-astra`              | high-quality | $10.00 | $1.00   | $50.00  |
| `gemini/gemini-4-argon`          | high-quality | $2.00  | $0.10   | $10.00  |

Gemma is self-hosted, so its rates are an estimate of what the deployment costs rather than
a published price: Baseten's H100 rate divided by an assumed throughput. Cached prompt tokens
are billed once, at the cached rate, not twice.

DeepSeek's models reason by default, which spends the whole token budget and most of the
latency before the first word of the answer. The provider turns thinking off through the
chat template, since that is the wrong trade for a live conversation; `Options.Thinking`
turns it back on and the reasoning then arrives as `ReasoningDelta` events, separate from
the answer.

Speech to speech has three shortcuts of its own. `sts-fast` is the model quickest to answer,
pinned to `gemini/gemini-3.8-live`; `sts-vision` is a conversation the model can watch as
well as hear, pinned to `openai/gpt-realtime-2.1`; and `sts-duplex` requires a model that can
listen while producing audio. Realtime transport, barge-in or asynchronous tools alone do
not establish full duplex. A native speech model owns transcription, turn-taking and speech,
so it takes the place of the separate STT, conversation LLM and TTS pipeline. A session asks
for one target rather than three, and the terms it can be asked for are its own:
`semantic_turns` for a turn detector that reads the words, which only OpenAI has;
`tools` and `text`, which Qwen lacks; `endpointing` for a silence timer whose length can be
set; and the two transcripts. Frames are an input modality, gated by `input_modalities`,
not a term. Which models the shortcuts reach, and what each is billed at:

Set a managed voice agent's speech target and optional skills subagent in `agent.yaml`:

```yaml
sts: sts-duplex
subagent: gemini/gemini-3.8-flash
```

The subagent is an independent LLM router target for delegated reasoning and skills; it can
name another configured provider/model or alias. It does not choose the speech provider's
own reasoning backend. GPT-Live uses OpenAI's Responses backend, which defaults to
`gpt-5.6-terra`.

A named STS router config sets defaults for direct router clients. For example, `routers/duplex.yaml`:

```yaml
name: duplex
sts:
  target: sts-duplex
  overwrites:
    openai:
      backend_model: gpt-5.6-terra
```

Sync the directory first with `client.SyncRouters(ctx, "routers")`, then open that config with
`client.Router("duplex").STS().Realtime(ctx, nil, nil)`.
This named router config does not configure a managed agent; its `agent.yaml` independently
chooses `sts` and the optional `subagent`. For a direct `/v1/sts/stream` start frame, pass
provider-specific settings in `sts.overwrites`:

```json
{"target":"openai/gpt-live-1","sts":{"overwrites":{"openai":{"backend_model":"gpt-5.6-sol"}}}}
```

`backend_model` is an OpenAI model ID for the Responses backend, not a provider/model routing
target. The agent's `llm` field and its optional `subagent` do not select that backend.

| Model                                     | Tier         | Billed by                                   |
| ----------------------------------------- | ------------ | ------------------------------------------- |
| `gemini/gemini-3.8-live`                  | low-latency  | tokens: $3.00 in, $12.00 out per million     |
| `gemini/gemini-3.1-flash-live-preview`    | low-latency  | tokens: $3.00 in, $12.00 out per million     |
| `gemini/gemini-3.8-live-extended-thinking` | high-quality | tokens: $3.00 in, $12.00 out per million    |
| `xai/grok-voice-3.0`                      | low-latency  | the caller's audio: $4.80 an hour            |
| `xai/grok-voice-think-fast-2.0`           | low-latency  | the caller's audio: $3.00 an hour            |
| `qwen/qwen3.5-omni-plus-realtime`         | low-latency  | the caller's audio: $0.03 an hour            |
| `openai/gpt-realtime-2.1`                 | high-quality | tokens: $32.00 in, $0.40 cached, $64.00 out  |
| `openai/gpt-realtime-2`                   | high-quality | tokens: $32.00 in, $0.40 cached, $64.00 out  |

The vendors price audio in, audio out and text at different rates, and where a vendor reports
its tokens by modality (OpenAI Realtime, Gemini Live) each is priced at its own rate in
`prices/vendor.json`. A reply is one row, settled when the model finishes it; the caller's audio
since the last reply is the row's `audio_ms`, and the tokens are what the vendor reported.
`stssuite` is what every one of these models is held to on a real call, the way `sttsuite`
and `ttssuite` hold a transcriber and a voice. `.claude/skills/router-sts` is the vocabulary.

Gemini is reached over Google's OpenAI-compatible endpoint, so it needs no implementation of
its own. Every Gemini 3 model thinks and none of them can be told not to, so the provider
pins the effort to the model's floor rather than letting a conversation wait on its own
default; `Options.ReasoningEffort` raises it for anything off the live path. That floor is
`minimal` up to 3.5 and `low` from 3.8 Flash, which made thinking a level and dropped
`minimal` with it, so asking for it there is refused here rather than 400ing mid-turn.
Gemini bills thinking as output, which is the other reason a live turn asks for the floor.
Google reports it as a token count rather than streaming it, so there are no
`ReasoningDelta` events to separate out.

### Transcribers, and what Gemini does differently

| Model                                                      | Languages | Per audio hour |
| ---------------------------------------------------------- | --------- | -------------- |
| `cartesia/ink-2`                                           | en        | ~$0.54         |
| `deepgram/flux-general-en`                                 | en        | $0.276         |
| `deepgram/flux-general-multi`                              | 12        | $0.276         |
| `elevenlabs/scribe_v2_realtime`                            | 57        | $0.39          |
| `gemini/gemini-3.5-transcribe-live`                        | 85+       | ~$0.54         |
| `grok/grok-stt`                                            | 25        | $0.20          |
| `inworld/inworld-stt-1`                                    | 30        | $0.15          |
| `microsoft/MAI-Transcribe-2-Streaming`                     | 60        | $0.54          |
| `muse/muse-voice-transcribe-1.0`                           | 25        | $0.18          |
| `parakeet/parakeet-tdt-0.6b-v3`                            | 25        | $0.079         |
| `together-nemotron/nvidia/nemotron-3-asr-streaming-0.6b`   | en        | $0.09          |
| `together-nemotron/nvidia/nemotron-3.5-asr-streaming-0.6b` | 28        | $0.27          |
| `together-parakeet/nvidia/parakeet-tdt-0.6b-v3-realtime`   | 25        | $0.21          |

The two Parakeets are the same weights in two places: `parakeet` is our own Baseten
deployment and `together-parakeet` is Together's serverless endpoint. They are separate
providers because the bill and the pager are not shared, so routing can pick between them
and fail over from one to the other.

The two Nemotrons share a provider instead, because they are two models on one endpoint
rather than one model in two places. Nemotron 3 ASR is English-only and is what NVIDIA
recommend for an English call; Nemotron 3.5 ASR is the multilingual extension of it,
covering 40 language-locales from the same checkpoint and detecting which one is spoken
rather than being told. The 28 codes it declares are NVIDIA's transcription-ready and
broad-coverage tiers; their third tier needs fine-tuning before it transcribes, so a call
in one of those languages is not routed here. Both speak the same realtime protocol as
`together-parakeet`, and their deltas restate the utterance rather than adding to it.

Muse Voice Transcribe is configured by its opening frame rather than by a header or a
query string, credentials included, so a rejected key arrives as an error event on the
socket rather than as a failed dial. It finds turn boundaries itself, which is what
`ENDPOINTING`, the mode the provider defaults to, asks for; `PUSH_TO_TALK` leaves them to
the caller and settles on the flag the transcript carries rather than on `speechComplete`.
Turns can overlap, so the provider keys its bookkeeping on the `turnId` the server sends
rather than on the order the frames arrive in.

Ink 2 is the second model here that decides where a turn ended itself, and Cartesia publish
a migration guide from Flux to it because they are much the same shape of protocol. The
three thresholds that move that boundary are reachable through `overwrites`, not as shared
terms: Cartesia's own guide says theirs and Deepgram's do not mean the same thing at the
same value. Closing the socket transcribes whatever the server was still holding, so the
tail of a caller who was cut off is not lost.

Inworld STT 1 and Scribe v2 Realtime both take their configuration where it cannot be
changed afterwards — Inworld in an opening `transcribeConfig` frame, Scribe in the query
string — and both send audio as base64 inside a JSON frame rather than as binary. Scribe has
no control frame at all: committing the tail of a call means sending a chunk with the commit
flag set, which is why the provider sends a fiftieth of a second of silence to carry it. It
also does not start transcribing until it has about two seconds of audio, so its first
interim transcript lands later than the others' while the turn still settles as quickly.

Muse and Grok are the only two of these that can tell one voice from another. What that is
for is the microphone with more than one person at it: a track says who joined the call,
which is the wrong answer for everybody else in the room with them, so the label is what
lets the agent leave a bystander's question alone. It is a label the session made up rather
than an identity, and the transcribers that cannot hear the difference leave it empty.

Grok names the voice on every session, because it costs nothing to ask for. Muse does it
only when a request asks, because its `DIARIZATION` mode looks for a speaker change rather
than a clean end of speech: measured on the same clip it settles a turn in about 2150ms
where `ENDPOINTING` takes about 770ms, and that second and a half is time the caller spends
waiting. Both declare `diarize` and both honour it, so asking narrows an English call to
those two rather than serving it unlabelled from one of the others.

Deepgram, Cartesia, ElevenLabs, Grok, Inworld, Muse, Parakeet and Nemotron are speech
recognisers. Gemini 3.5 Transcribe
is a Gemini model that happens to be listening, reached over the Live API's
`BidiGenerateContent` socket with the talking half turned off, and that difference shows
in three places.

It writes down what you meant rather than what you emitted: filler words go, a
self-correction resolves to the correction, and the text arrives punctuated. That is worth
having on a call and wrong for a compliance transcript, and it is not a setting.

It has no field for a keyterm or a language hint — `AudioTranscriptionConfig` is an empty
message. Both are passed as a system instruction instead, which is the only lever the API
gives and is a request rather than a constraint. Deepgram's `keyterm` is the stronger tool
if the vocabulary matters more than the phrasing.

It sends the words in pieces that append, so a chunk is a `delta` and the turn boundary the
server reports is what settles it. The final restates the whole utterance, which cadence
treats as the transcriber settling rather than as the caller saying it twice.

Billing is per token, not per hour: audio in at $3.50/1M against 25 tokens a second, text
out at $21.00/1M. The router config carries Google's own blended figure of about $0.009 an
audio minute, so a busy call costs more than the table suggests and a quiet one less.

## Run

```bash
# The API
ROUTER_POSTGRES_DSN=postgres://... ROUTER_REDIS_ADDR=localhost:6379 \
  go run ./cmd/router
# Say something
go run ./cmd/say -text "Hello from the router."

# Or type lines and hear them
go run ./cmd/say -target multilingual-low-latency -language es

# Ask a model something
go run ./cmd/chat -text "Name three primary colours."

# Or hold a conversation, which keeps its history between turns
go run ./cmd/chat -target en-high-accuracy

# Join a Stream call and talk to it. A browser opens on a link that joins the same call,
# which -demo=false turns off when the caller is joining from somewhere else.
go run ./cmd/agent -call my-call

# Sprint 6 stack: Gemma speaks, Sol handles the hard parts
go run ./cmd/agent -call my-call \
  -stt parakeet/parakeet-tdt-0.6b-v3 \
  -tts fish/s2-pro \
  -llm gemma/gemma-4-26B-A4B-it \
  -subagent openai/gpt-5.6-sol

# Label what a session costs, so spend can be broken down later
go run ./cmd/agent -call my-call -tag project=support -tag environment=dev

# Give an agent a phone number
go run ./cmd/phone vendors
go run ./cmd/phone search -vendor twilio -country US -area 512
go run ./cmd/phone buy -vendor twilio -number +15125551234 -tag project=support
go run ./cmd/phone attach -number +15125551234 -call support-line

# Or ask every vendor at once, for somewhere specific with the features you need
go run ./cmd/phone search -country US -state CO -type local \
  -feature hd_voice -feature emergency -limit 5

# Call somebody, on the terms the call needs, and join the call it prints
go run ./cmd/phone dial -from +15125551234 -to +15550001111 \
  -call-id support-line -ring-timeout 20s -digits ww1234#

# Let the agent hand a caller to a human, or answer a menu on a call it placed
go run ./cmd/agent -call support-line -number +15125551234
go run ./cmd/agent -call outbound-1 -number +15125551234 -vendor-call v3:abc -navigating

# Or do either by hand
go run ./cmd/phone transfer -from +15125551234 -to +15550002222 -call support-line
go run ./cmd/phone press -vendor telnyx -call-id v3:abc -digits 1
```

Each utterance `cmd/say` finishes prints which provider served it, the wait for the first
audio, how much speech came back and what it cost:

```
elevenlabs/eleven_flash_v2_5  first audio 158ms  audio 2.7s  55 chars  $0.002750
```

`cmd/chat` prints the same shape of line per turn, in tokens rather than characters:

```
deepseek/DeepSeek-V4-Flash-0731  first token 356ms  20 in  13 out  $0.000005
```

## The agent

`internal/agent` is the Go counterpart of the Python `Agent`, built from the three routers
plus a target each. One conversation is one LLM session and one voice session, but a
transcription session per participant, because a speech-to-text stream is bound to a single
speaker.

```mermaid
flowchart LR
  edge["Edge audio 16k mono"] --> stt["STT session per participant"]
  stt -->|"transcript revisions"| cadence["Cadence"]
  cadence --> flow["Fast flow controller"]
  flow -->|"respond or clarify"| conv[Conversation history]
  flow -->|"stop, shorten, continue"| floor["Speech floor"]
  conv --> harness["Harness"]
  harness --> llmSession["LLM session"]
  llmSession -->|"text deltas"| harness
  harness -->|"speech"| chunker["Sentence chunker"]
  chunker --> ttsSession["TTS session"]
  ttsSession -->|"PCM chunks"| out["Edge audio track"]
```

Three decisions are worth knowing about:

- **Cadence, not turn detection, decides when to act.** Transcript revisions are debounced
  per participant, then a separate fast-model session decides whether to wait, ignore
  background speech, respond, or clarify. A provider final is metadata rather than the
  response trigger; new words cancel a stale decision.
- **The reply is spoken sentence by sentence.** A model emits a few characters at a time,
  and a voice given two words at a time pauses in the wrong places. A streaming voice takes
  a turn's sentences as deltas of one utterance, so one turn stays one billed synthesis; a
  voice that cannot take deltas gets one final request per sentence.
- **Overlap is a floor decision.** A correction stops the model and voice, a related addition
  shortens the current answer after speech already queued, and an acknowledgement lets it
  continue. Audio from an abandoned turn is still dropped at publication.

Those three decisions are where a call goes wrong, so `ROUTER_LOG_LEVEL=debug` narrates
them: every transcript revision, when the words held still, what the flow controller was
asked and what it answered, and why the agent then spoke, waited, murmured, queued the turn
or stopped mid-reply. A quiet agent is usually one of `ignore`, `wait` on repeat, or a turn
queued behind speech that never settled, and each of those says so.

```
transcribed provider=deepgram participant=user-1 mode=replacement text="what's the weather"
the words held still, asking whether to answer them candidate=turn-1787 waited=352ms
the flow controller decided candidate=turn-1787 disposition=respond floor=continue
answering candidate=turn-1787
```

`Edge` is four methods (`Join`, `Audio`, `PublishAudio`, `Leave`), which is what lets the
whole flow be tested in-process against a loopback rather than only against a real call.
`streamedge` is the real one: it joins over the `getstream-go-webrtc` SDK, subscribes
to the audio of everyone else in the call (joining subscribes to nothing on its own), decodes
inbound Opus to 16 kHz mono, and encodes the agent's speech back to 48 kHz Opus.

## The harness

The model on the live path is chosen for how quickly it starts talking, which is not the
same as how well it thinks. `internal/harness` is what stops that being a trade made on
every sentence: the fast model can hand the hard part to a slower one and go on talking
while it runs.

```mermaid
flowchart LR
  stt["Transcript cadence"] --> controller["Flow controller"]
  controller -->|"respond or clarify"| h["Harness"]
  h -->|"reply"| fast["Fast LLM session"]
  fast -->|"deltas"| filter["Directive filter"]
  filter -->|"speech"| chunker["Sentence chunker"] --> tts["TTS"]
  filter -->|"ask for help"| tasks["Task manager"]
  tasks --> sub["Subagent LLM session"]
  sub -->|"result"| h
  h -->|"cancel superseded"| tasks
  sub -->|"private summary"| compact["History compaction"]
```

A skill is a name, a line telling the fast model what it is for, and the instructions the
subagent answers under. `skills.yaml` is embedded, and `HARNESS_SKILLS` or `-skills`
replaces it, the same way `router.yaml` and `phone.yaml` already work. An agent config's own
skills are offered by name and description only, and their instructions are read from the store
when one is used, so an edit reaches the next use rather than the next session.

The harness is agent config, never session config: `harness` (only `default` today),
`subagent`, `sandbox` and `skills` are set on the config or in `agent.yaml`, and
`createSession` does not take them. Only a voice agent names a `subagent`: a text agent
runs everything, skills included, on its `llm`, and a config that gives one a subagent
is refused. A client that wants a sandbox of its own declares it as a
tool instead.

- **The model asks for help mid-sentence.** It writes `<ask skill="think">…</ask>` into its
  reply. A streaming filter takes it back out before the reply reaches the voice, so the
  caller hears "let me check that" and never the request. Everything that cannot yet be a
  tag is released immediately, because the caller is listening to the gap.
- **A model that has tools is told how to use them.** The reply model is given a short block
  after the agent's own instructions whenever the harness offers tools: before calling one,
  say one short sentence that opens with a brief hold phrase ("One moment,") and runs straight
  on, with no full stop between, into what the operator's instructions ask to be said before
  acting, such as reading the caller's details back, or else what the agent is doing. The
  phrase belongs there and nowhere else, never a sentence of its own, never in place of a
  required read-back and never after the call or the result: a read-back takes seconds to say,
  so a phrase after it comes after the wait it was meant to cover, and a one-word sentence
  before it leaves a pause a caller's interruption falls into. After a result the model
  answers from it without another one. It then calls the tool in the same turn once every
  argument it requires is known, without collecting optional ones first or asking leave for
  what the caller just asked for; takes a name or value as the caller gave it, since a surname
  is a name; after a result, calls the tools the request still needs straight away and without
  another hold phrase, together when they do not need each other, rather than a sentence and a
  model turn per tool; and passes bare values. It
  names no tool and defers to the operator's own instructions and a tool's approval setting
  wherever confirmation is required first. A reply carrying a colleague's question is offered
  no tools and is not told about them. The agent's own filler and a tool's `pre_speech` stay
  what they were, for a model that reached for a tool without a word: a turn that said
  something before its call is given neither, so a wait has one hold phrase.
- **A task is a completion, so cancelling one is a targeted interrupt.** `Interrupt` on
  `llm.LLM` takes completion ids, which is what lets a conversation abandon stale work
  without stopping the reply being spoken. Work is abandoned when a newer request for the
  same skill or a revised conversational premise supersedes it, when the model writes
  `<drop skill="…"/>` because the caller moved on, when its deadline passes, or when the
  call ends.
- **Answers come back as a turn nobody asked for.** The caller was told an answer was
  coming, so it arrives without them asking again. If the agent is mid-sentence the answer
  waits for it to finish rather than talking over itself. A subagent that cannot answer
  without knowing something replies `NEED: …`, which becomes a question in the agent's own
  words.

Subagent completions go through `llmrouter` like anything else, so what the thinking costs
lands in `requests` with the same failover and cost tags as the talking.

An agent whose config names a sandbox gives the subagent `run_code`, and on a call the model
holding the conversation none: running code takes seconds that a call does not have, and the
subagent has already left the live path. A text session has no subagent, so the model holding
it is offered `run_code` itself (see [Conversations held in writing](#conversations-held-in-writing)). Code the subagent
writes runs in Daytona, its output comes back as a tool result, and the same task is put
again, up to four rounds and always inside the skill's own deadline. The subagent may also
`lookup` and `search` when the agent has a knowledge base and a search target, knowledge base
first. One sandbox is created the first time code actually runs and released when the session
ends.

`sandbox_options` on the config, or in `agent.yaml`, says how that sandbox is built and how
long code may run in it: an `image` with Python to start from, `setup` commands run on top of
it once, `timeout_ms` for one run (30 seconds by default, at most 30 minutes), and `cpu`,
`memory_gb` and `disk_gb`. Any of them builds an image, which Daytona keeps, so only the
first sandbox from a given setup waits for the build. A run is still bounded by its skill's
deadline, so a skill that renders for minutes says so in its own.

Code in the sandbox reaches nothing outside it by default. `network: true` opens the
internet, and `allowed_domains` or `network_allow_list` (CIDR ranges), one or the other,
names the only places it may go, with or without `network`. These need no image of their own.
Daytona organizations on tiers 1 and 2 cannot set either list per sandbox and refuse to create
one that does; blocking everything, the default, works on every tier.

`run_code` takes `files` as well as `code`: paths the program writes that the person should
get, at most four of up to 20 MiB each. On a persistent text conversation each file is
uploaded to the channel as the agent, `task_settled` lists them (`name`, `mime_type`, `url`,
`size`), and the reply that settles the work carries them as Chat attachments, an image
inline and anything else as a file, so they are there when the conversation is reopened.
Without a channel the subagent is told there is nowhere to show them.

Long histories are compacted privately on the thinking session when the prompt is large and
either it has filled 80% of the conversation model's `context_window` in `router.yaml`, or the
provider's reported cached-token ratio has fallen below half. The result replaces
only the unchanged old prefix; recent turns stay verbatim and a late summary cannot overwrite
newer conversation.

### Listening and talking at the same time

Cadence and floor control are always part of the agent. `-backchannel` defaults on and makes a
short listening noise during long speech or delegated work; pass `-backchannel=false` to turn
it off. `-min-confidence` additionally makes the agent clarify a doubtful transcript. The
flow controller also asks for clarification when the words are clear but the intent is not.

### Voices that act a direction

Some voices perform bracketed directions such as `[laughs]` or `[whispering]` rather than
reading them out. A provider says whether it does, and only then is the model told it may
write them. What it writes is stripped before it reaches the transcript or the
conversation's history, so a direction is heard and never read.

Three providers perform them, and they are not interchangeable.

`elevenlabs/eleven_v3_conversational` acts any bracketed direction, over the Text to
Dialogue socket rather than the streaming one. It is a different bargain from
`eleven_flash_v2_5`: latency is roughly a second to first audio against about 75 ms, and
an open connection holds a dialogue session from a pool separate from the standard
concurrency limit, so a deployment routing calls to it needs headroom of a different kind.
The protocol has no context ids and no cancel frame, so audio is attributed by flush order
and a barge-in reopens the socket.

`breeze/breeze-tts-2` acts four: `[laugh]`, `[sigh]`, `[cough]` and `[clears throat]`. Its
prompt says so, because telling a model any direction works when only four do would have
it write directions that get spoken. In exchange it takes the other half of the
instruction — how the line should sound, or who should say it — which no other provider
here does. Two languages only, English and Chinese, and it runs on our own Baseten
deployment, so see `deploy/breeze-tts-2/README.md` before routing anything at it: the
licence and the single-request engine both matter.

`fish/drama-3` acts any bracketed direction, written the way you would brief an actor:
`[calm, measured storyteller voice]` as readily as `[sighs]`. It is Fish's hosted API,
the same `POST /v1/tts` as `fish/s2-pro` with the `model` header changed, and the default
for the `fish` provider. Like every Fish model it synthesises a whole sentence per request,
so it answers later than the socket voices. `fish/s2-pro` reads a bracket out as words.

### A voice described rather than chosen

Every other provider takes a voice id, or one of your own voices written `custom:<name>`,
which is resolved per provider (one that was never given it is skipped). Breeze takes
prose: "a warm, thoughtful young woman with a calm, reflective delivery" designs a voice
with no reference audio at all, and the same field alongside a reference clip steers
delivery while the clip keeps the identity. So
`-voice` means something different depending on who is listening, and a voice id from
another provider is meaningless here rather than merely wrong. A `custom:` voice never
lands on Breeze, which has no cloner (below).

That is also why `breeze` registers no cloner in `internal/tts/voices`, the same as
`s2pro`: a customer's own voice is a clip plus a transcript per request, not something
registered upstream and given an id. The router skips it for a call asking for one.

### Directions are written in square brackets, whoever is listening

Breeze's own syntax for an English vocal event is parentheses — `(sigh)` — but the agent
writes `[sigh]` for every provider, because parentheses are ordinary punctuation that a
reply may use for something else. The deployment translates on the way into the engine,
which is the point where an utterance split across text deltas is whole again. A Chinese
event such as `[叹气]` is already in brackets and passes through untouched.

## Sessions, for callers outside this process

Everything above is driven by `cmd/agent` on a command line. `internal/session` is the same
wiring offered over HTTP, so a Python process can have this service hold a conversation on
its behalf: the spec's fields are `cmd/agent`'s flags, and the process that used to be
started per call becomes a session in a process that is already running.

| Endpoint                                | What it does                                    |
| --------------------------------------- | ----------------------------------------------- |
| `POST /v1/agents/sessions`              | Join a call. Returns once the agent is listening |
| `GET  /v1/agents/sessions`              | The customer's sessions, newest first            |
| `GET  /v1/agents/sessions/{id}`         | One of them                                      |
| `DELETE /v1/agents/sessions/{id}`       | Leave the call                                   |
| `POST /v1/agents/sessions/{id}/say`     | Speak, without going through the model           |
| `POST /v1/agents/sessions/{id}/respond` | Answer something, as though it had been said     |
| `POST /v1/agents/sessions/{id}/interrupt` | Abandon the reply being spoken                 |
| `POST /v1/agents/sessions/{id}/instructions` | Change what the agent is, from the next turn |
| `GET  /v1/agents/sessions/{id}/events`  | WebSocket: everything the agent does             |
| `GET  /v1/{modality}/stream`            | WebSocket: one modality, for a pipeline elsewhere |
| `GET  /v1/agents/socket`               | WebSocket: a voice session with no call, its audio on the socket |

```bash
curl -s localhost:8080/v1/agents/sessions -H 'X-Customer-Id: acme' \
  -H 'Content-Type: application/json' -d '{
    "call_id": "demo-1",
    "instructions": "Keep your replies short.",
    "llm": "llm-fast",
    "tags": {"project": "support"},
    "memory": {"user_id": "222"}
  }'
```

Two things travel in both directions. The socket carries out what the agent heard, said and
decided, and carries back `say`, `interrupt` and `close`. And it carries `tool_call`: the
caller's own functions live wherever the caller is, so the model asks for one here, the
answer comes back from there, and a caller that does not answer inside the tool timeout
leaves the model to carry on without it rather than leaving the call silent.

Sessions are keyed by customer. A session id that exists but belongs to somebody else is
reported as not existing at all, since a forbidden would confirm it was real.

### Conversations held in writing

`"text": true` and no `call_id` holds the conversation in writing instead. No call is
joined, nothing is transcribed and nothing is spoken, so neither speech target is used:

```bash
curl -s localhost:8080/v1/agents/sessions -H 'X-Customer-Id: acme' \
  -H 'Content-Type: application/json' -d '{"text": true, "config_id": "..."}'
```

Everything between hearing a question and answering it is unchanged, which is the point. A
text session has the same instructions, the same skills and the same knowledge base as a call
would have had, so an agent can be built and argued with in writing before anybody rings it,
and a documentation agent or a support chat is the voice agent with the voice left off. The
one difference is who runs a skill: nobody is waiting on a voice, so the model holding the
conversation loads the skills its config names itself, with `load_skill`, and keeps every
tool while it follows one. It takes answers from a matching skill first, then the knowledge
base (`lookup`), then the web (`search`). A text agent with a sandbox runs code itself with
`run_code`, a loaded skill's scripts written under `skills/<name>/` before it runs, and the
files it hands back ride on `tool_ran` and the reply.

It is driven over the session's own socket: `respond` goes in, `response_delta` and
`responded` come back, along with `looked_up`, `tool_ran`, `delegated` and `task_settled` as
the agent works. `say` and `interrupt` do nothing useful here, since there is nothing being spoken to
interrupt.

### Conversations held over a socket

`GET /v1/agents/socket` holds a voice conversation with no call to join: the audio travels
on the socket itself. It is for callers that are neither a browser nor a phone, such as a
benchmark's simulated caller, and everything between hearing and answering is the agent's
own, the cadence, the flow controller, speculation and barge-in included.

The first frame is `{"type": "start", "sample_rate": 16000, "session": {...}}`, where
`session` is what `POST /v1/agents/sessions` takes and `call_id` may be left out. The
answer is `{"type": "session", "session": {...}, "sample_rate": 16000}`. From then on
binary frames are PCM16 mono at that rate both ways: the caller's audio in, and the agent's
speech out in 20 ms frames at the pace it is heard on a call, so a reply the caller cuts
into stops within a frame. `{"type": "cleared"}` says speech already sent was thrown away
because the caller cut in. Tool calls and everything else the conversation does go over the
session's events socket as they would for a call.

The session lasts as long as the socket. Closing it, or sending `{"type": "stop"}`, ends
the conversation, and a conversation that ends closes the socket. Unlike the events socket
it cannot be relayed, because the audio travels on the connection itself: the session is
the node the socket opened on, and that is where it stays.

### More than one node

A session lives in one process's memory, and a load balancer has no reason to send a
caller to the process holding theirs. Two things make the deployment answer anyway, and
both need `redis.addr`:

- The events socket is relayed. `internal/relay` publishes a session's frames on one
  Redis pub/sub channel and the watcher's commands on another, and the node holding the
  socket attaches a real watcher on the node holding the conversation. Every node is sent
  every session's events, so each keeps a cuckoo filter of the owners it holds sockets for
  and drops the rest before they cost anything.
- Everything else is carried to the node that can answer it. `internal/node` keeps a
  register in Redis of which node is running which session, renewed while it runs, and a
  request naming a session that is not here is handed to the node that is by gRPC. Peers
  are reached over h2c on the API's own port, so there is no second port to open between
  them, and the node answering runs its own handler: it authenticates and authorizes the
  caller itself rather than taking a peer's word for either. Since anyone reaching that
  port can speak gRPC to it, a peer presents a credential derived from the key encryption
  keyring (`ROUTER_AUTH_KEK`, `ROUTER_AUTH_KEK_V<n>`), and a node accepts the one of any key
  it holds, so a rotation needs no pause. A node with no keyring refuses every peer, and
  every node holding one refuses it.

That register also answers who is writing as an agent, which is what an arriving chat
message names. Without it a message landing on the wrong node finds nothing running and
is handed to a worker, which is a second agent writing into a conversation the first is
already answering in.

A node that dies leaves its claims behind until they expire, during which a request for
one of its sessions is answered with a 502 rather than a 404: the conversation existed and
the node holding it has gone, which is a different thing from a session that was never
there.

### Shards

One node with one Postgres and one Redis is a whole deployment, and needs nothing below:

```yaml
postgres:
  dsn: postgres://localhost/router
redis:
  addr: localhost:6379
```

A hosted deployment runs one shard per region. Every shard shares a global Postgres and
Redis (tenancy, agents, connectors, phone numbers, the request log, quotas, the node
register and the worker bus) and keeps its own for what it runs: calls, sessions,
responses, requests and their rollups in Postgres, provider health and the session relay
in Redis. Any shard serves any app.

```yaml
shard: us-west1
postgres:
  dsn: postgres://global/router
  shard_dsn: postgres://usw1/router
  shards: # every shard's, this one's included
    us-east4: postgres://use4/router
    us-west1: postgres://usw1/router
    europe-west4: postgres://euw4/router
redis:
  addr: global:6379
  shard:
    addr: usw1:6379
```

Writes go to this shard's database. Reads of one call, session or recording look in every
shard, and lists merge every shard's rows, so a session carried on in another region is
still one session. Migrations run on every database. A socket for a session held in
another shard is proxied to the node holding it, since that shard's relay is not this
one's. Empty shard settings fall back to the global ones, which is the single-node setup.

Two things are approximate across shards: percentiles merged from several shards' rollups
are weighted by request count rather than recomputed, and a data move replicates the
changes of the global tables only, so a customer's calls and sessions are moved by export.

## Agents that are configured rather than spelled out

A session can be created from a stored configuration instead of a caller repeating the
whole spec every time. `config_id` on `POST /v1/agents/sessions` loads one and everything
inline in the request overrides it, so a config can be reused and one call still changed.

| Endpoint                                    | What it does                                |
| ------------------------------------------- | ------------------------------------------- |
| `/v1/agents/configs`                        | Named agents: models, voice, instructions, skills, knowledge |
| `/v1/agents/skills`                         | What the fast model may hand to the slower one |
| `GET /v1/agents/calls`                      | Calls that happened, running ones included   |
| `GET /v1/agents/calls/{id}`                 | One call, with what a model made of it       |
| `GET /v1/agents/calls/{id}/transcript`      | What was said                                |
| `GET /v1/agents/calls/{id}/timeline`        | Each exchange, said and measured together    |
| `GET /v1/agents/calls/{id}/events`          | What the conversation decided, and why       |
| `POST /v1/agents/knowledge`                 | Fill a knowledge base from documents         |
| `/v1/agents/knowledge/urls`                 | Pages a knowledge base is kept filled from   |
| `/v1/agents/campaigns`                      | Lists of people to ring, and how far they got |
| `/v1/agents/scheduled-tasks`                 | Things the agent does later for a user, once or on a repeat, and what came of each occurrence |

**Skills are named rather than spelled out.** A config carries skill names, and they are
resolved once when the session is created: the customer's own rows, or the built-in
`think`, `recall` and `explain`. A name nothing defines is refused rather than dropped,
because a model offered a skill nobody implements is worse than one offered none. The
runtime lookup is unchanged.

**Knowledge is what the business wrote down.** With `TURBOPUFFER_API_KEY` set, a config's
`knowledge_namespace` gives the model a `lookup` tool over full-text search, so a caller's
question about prices or opening hours is answered out of the handbook rather than guessed
at. Each search is a `requests` row with modality `knowledge`, like everything else that
costs money: a `lookup`, priced per lookup, carrying the bytes turbopuffer billed it as
searching (at least 1.28 GB) and returning, and each write or delete request a `write` of the
bytes written. What each cost turbopuffer is kept as `attributes.provider_cost_micros`.

Storage is sampled hourly by `router usage storage`: one `storage` row per knowledge base and
hour, of its size as `byte_hours`, owned by its app under `job:knowledge_storage`. By default
it records the last hour to have ended, so a retry within the next hour bills the same one;
`--hour` records another, at the size the base holds now, since turbopuffer keeps no past
sizes. Run exactly one such job per turbopuffer key and region, not one per shard or router:
it reads every namespace the key holds, and skips those of apps this deployment has never
seen, which belong to another deployment sharing the key. A base it cannot read fails the run
and alerts as `storage_missed`, naming the hour to run again.

**Search is what is true today.** Neither the model nor the handbook can answer what the
traffic is doing, whether a place is open or what a score is: one was trained months ago and
the other was written down once. A session gets a `search` tool wherever a provider can be
built for it, and an agent asked about any of those looks it up instead of saying it cannot
check.

Search is routed the way the three model modalities are. A config's `search` names a target,
which is a `provider/model` or one of the same capability shortcuts the others use, and the
router ranks the candidates, fails over past a provider whose key is missing, and files one
`requests` row per search under modality `search`. There are three providers and they differ
more between their own modes than they do from each other, which is what the models select
between:

| Target                | What it does                                                     |
| --------------------- | ---------------------------------------------------------------- |
| `exa/fast`            | Exa's index, no crawl and no model: the quickest sources          |
| `perplexity/search`   | Perplexity's ranked index, results only                           |
| `tavily/basic`        | Index plus a written summary in the same round trip               |
| `exa/auto`            | Exa choosing between its neural and keyword indexes               |
| `tavily/advanced`     | Crawls the pages it finds rather than trusting the snippet        |
| `perplexity/sonar`, `perplexity/sonar-pro` | A model reads the pages and writes the answer |

A session that names nothing gets `search-fast`. Sonar is a target rather than a fork:
because it is a language model in the middle of a conversation, whether it is worth its
latency is a routing decision.

The answer lands in the conversation as a tool result, which is what puts it in front of the
subagent too: a caller who asks for the best route given the closures gets the fast model
finding out what the closures are and the subagent reasoning over them. Nothing extra is
configured for that, and nothing is offered at all without a key for something.

**A knowledge base can also be filled from URLs.** Posting a document happens once; a url is
a subscription, because the page behind it changes and nobody re-posts it. With
`EXA_API_KEY` set alongside turbopuffer, a database and Redis, `/v1/agents/knowledge/urls`
fetches the page, turns it into markdown and cuts it into passages the same way a document
is. The read is an [asynq](https://github.com/hibiken/asynq) task on Redis, run by a worker
inside the router: the request answers with the page `pending`, and a read that fails is
retried three times before the page is marked `failed`.

```bash
curl -X POST localhost:8080/v1/agents/knowledge/urls \
  -H "X-Customer-Id: acme" -H "Content-Type: application/json" \
  -d '{"namespace":"docs","url":"https://example.com/pricing","refresh_hours":24}'
```

Each row records when the page was last read successfully, what it was called and how many
passages it became. That last number is what makes removing it exact: passages are keyed by
the url and a position, so deleting a subscription takes its passages with it, and
re-reading a page that got shorter leaves no orphans behind. A page that could not be
fetched is still stored, in the `failed` state with the reason on it, rather than refused
and forgotten.

A page with `refresh_hours` is read again once that many hours have passed since its last
read, failed or not. Every router looks once a minute for pages that are due and queues them;
the task id is the page, so two routers queuing one read it once. Without `refresh_hours` a
page is only read again when asked: `POST /v1/agents/knowledge/urls/{id}/index` queues a
read, and so does adding the page again.

`cmd/knowledge` fills one from files:

```bash
go run ./cmd/knowledge -customer examples -namespace docs ../docs ../README.md
```

A namespace belongs to a customer: turbopuffer stores it as `<customer>__<namespace>`, so two
apps that both say `default` never read each other's. `-customer` is the app id the agent
runs under.

`POST /v1/agents/knowledge` fills one from a caller that has no access to this machine's
disk, which is what an SDK pushing an agent directory needs. Documents are cut into passages
by the server, in `internal/knowledge/ingest`, so a file read off disk by the command and one
posted over HTTP are cut the same way and replace each other.

```bash
curl -s localhost:8080/v1/agents/knowledge -H 'X-Customer-Id: acme' \
  -H 'Content-Type: application/json' -d '{
    "namespace": "docs",
    "documents": [{"source": "pricing.md", "text": "# Pricing\n\nA call costs..."}]
  }'
```

Markdown is cut at its headings, a section too long to be one passage is cut again at
paragraph breaks, and each piece keeps the heading it was found under so it still says what
it is about when it comes back on its own. A section that is only a heading is skipped:
retrieving a title tells the model nothing. Passages are keyed by the file and the position
they came from, so reading a directory again after editing it replaces that file's passages
rather than leaving two versions of them to be found. `-dry-run` prints what would be
written without writing it.

Search is BM25 only. There are no embeddings here, and nothing to keep in step with a model
that generates them: documentation and handbooks are answered well by the words in them, and
a lookup on the answering path has to be fast.

**A call outlives the session that ran it.** Sessions live in a map in memory, which
answers what is happening now and nothing at all about last Tuesday. A `calls` row is
written when the agent joins and again when it leaves, both off the conversation's path.
What was said is not copied into Postgres: the transcript is already in Stream Chat, keyed
by agent, and the timings are kept in Redis for an hour. The timeline endpoint is those two
joined, so it shows a call's turns only while they are that recent. When the call ends, a short model pass over the conversation writes a summary and
a score from one to five onto the row, billed through the LLM router like anything else.

**Campaigns are the outbound half.** A campaign is one agent, one of the customer's
numbers and a list of people, each with instructions of their own that are added to the
config's. Starting one runs a loop that holds a semaphore sized to `concurrency`, places
each call at the vendor and creates a session for it; contacts are claimed in Postgres, so
a process that stopped mid-campaign resumes rather than starting again. Pausing stops
ringing new people and leaves the conversations already happening alone.

## Voices a customer brought with them

A config's `voice` is normally a name the chosen provider knows. It can also name one of
the customer's own voices, cloned from recordings they uploaded:

```bash
curl -s -X POST localhost:8080/v1/agents/voices \
  -H "X-Customer-Id: acme" -H 'Content-Type: application/json' \
  -d '{"name":"founder","description":"the one from the ad"}'

curl -s -X POST localhost:8080/v1/agents/voices/$ID/samples \
  -H "X-Customer-Id: acme" -H 'Content-Type: application/json' \
  -d '{"audio":"'"$(base64 < clip.wav)"'","filename":"clip.wav","transcript":"..."}'

curl -s -X POST localhost:8080/v1/agents/voices/$ID/prepare \
  -H "X-Customer-Id: acme" -H 'Content-Type: application/json' -d '{}'
```

**The voice is ours and the ids are theirs.** Preparing sends the recordings to each
provider that can be taught one and remembers what each called it back. A session names
our voice, and which id that means is worked out once the router has picked a provider —
which is the point, because the router fails over mid-call and an id ElevenLabs knows
means nothing to Cartesia. A provider that was never given the voice, or that refused the
recordings, is simply not chosen for a call that asks for it. Deleting a voice takes it off
every provider first, so a voice nobody can reach stops being billed for.

Recordings live in the bucket `ROUTER_VOICES_BUCKET_URL` names, not in Postgres, so they can be
re-sent to a provider added later without asking the customer for them again. Without that
variable the voice paths say so rather than half-working. Not every provider clones:
`s2pro` takes reference audio per session rather than registering a voice, so it is not
offered custom voices and the router routes around it for calls that want one.

## What is recorded

| Table                                    | One row per                                        |
| ---------------------------------------- | -------------------------------------------------- |
| `requests`                               | Unit of work: a turn, a synthesis, a completion, a memory call, a number bought |
| `stats_hourly`, `stats_daily`            | Bucket, modality, customer, provider and model      |
| `stats_tags_hourly`, `stats_tags_daily`  | Bucket, modality, customer and one cost label       |
| `turn_stats_hourly`                      | Hour, customer and agent: turns, interruptions, audio |
| `turn_latency_hourly`                    | Hour, customer, agent, leg and latency bin           |
| `phone_numbers`                          | Number held, kept after release because it was billed |
| `agent_configs`, `skills`                | Named agent, and a kind of work worth delegating    |
| `calls`                                  | Conversation that happened, with its summary and score |
| `campaigns`, `campaign_contacts`         | List of people to ring, and what became of each     |
| `scheduled_tasks`, `scheduled_task_runs` | Thing a user is to be reminded of or done for later, and each occurrence of it |
| `audit_log`                              | Change somebody made to the app's configuration     |

Four things fall out of this that are worth stating.

**Cost tags are the customer's own labels.** Any keys they like, up to sixteen per request,
carried onto every row a session produces. They get their own rollup tables rather than more
columns on `stats_hourly`, because a request carries a set of labels rather than one more
dimension: a request tagged `{project, environment}` is unrolled into both breakdowns. Ask
what drives spend with `GET /v1/{modality}/stats/tags?key=project`, or narrow the ordinary
stats with a repeatable `tag=project:support`. Filtering reads the raw request rows, since a
rollup bucket has already forgotten which labels its requests carried.

**A turn is not a request.** A request row measures one provider call. A turn measures what
the caller felt: the gap between finishing a sentence and hearing the answer start, which
spans three providers and the agent's own handling. Every leg is nullable, because a
realtime model that hears and speaks for itself fills only `roundtrip_ms`, and an interrupted
turn never reaches the legs that come after the interruption. A leg that never happened is
left out of the percentiles rather than counted as instant. A turn itself, and every decision
the conversation made, is kept in Redis for an hour and then let go; what lasts is each hour's
counts and a histogram of each leg in bins 5% apart, added to as each turn finishes, so
`GET /v1/turns/stats` reports percentiles within 5% of the true ones and a day is its hours
added up.

For a single call, `GET /v1/agents/calls/{id}/timeline` shows cadence settling, the
decision before the reply model, first reply text, TTS submission and first audio at the
edge. Its `model_calls` list keeps flow, reply and subagent requests separate, with TTFT
and full duration for each attempt. `speech_end_to_audio_ms` estimates the delay from the
last input audio using provider STT processing time; it cannot include network transit
or browser playback. `roundtrip_ms` and `speech_end_to_audio_ms` stop when publishing the
first chunk of the reply returns, which for a chunk longer than the 400 ms outgoing queue is
later than it could first be heard. `first_frame_queued_ms`, `first_audible_frame_ms` and
`speech_end_to_audible_ms` stop at the first frame being queued for the outgoing track and at
the track taking the first frame that is not silence; they are absent for an edge that does
not report them. The router logs call join and each ICE peer connection
independently; the subscriber can stay idle until someone publishes a track. The Python
accelerated agent records join duration. Set `log_latency=True` on `stream.Accelerated`
(or `VOICE_LATENCY_DAG=1` for `simple_voice_ai`) to print per-model timing and a console
DAG for each turn. The router records metrics regardless of this logging flag.
Model-call durations annotate the stages they overlap and are not added to the
speech-to-audio path.

**Memory and phone are recorded but not routed.** There is one memory store and one vendor
per number, so the provider and route paths do not serve those modalities while the
statistics paths do.

**No vendor is called without its usage being recorded.** The routers' sessions count what
they can see (the audio streamed to a transcriber, the characters sent to a voice, the
prompt and what came back from a model) and bill that, marked `usage_estimated`, when the
vendor says nothing; request-and-response work goes through `routing.Call`, and work billed
after the fact through `routing.Settle`. A session hands its vendor to nobody. A
speech-to-speech session also bills what its vendor reports on the side: GPT-Live's seconds
(`session_seconds`) and its backend model's responses (LLM rows with
`attributes.parent_session`), OpenAI Realtime's input transcription (a row of
`gpt-4o-mini-transcribe`), and the audio xAI bills and its typed turns (`audio_ms`,
`messages`). `internal/metered/guard_test.go`, part of `go test ./...`, fails a file that holds
a vendor-facing type or imports a vendor's package outside the routers, `internal/metered`
and the services that meter their own work, and a model vendor that runs no suite. So adding a
vendor is:

1. one package under its modality implementing the modality's interface, whose tests embed
   the modality's suite (`llmsuite`, `sttsuite`, `ttssuite`, `stssuite`, `searchsuite`,
   `detectionsuite`, `imagegensuite`, `decisionmodelsuite`), which checks it reports what
   it used;
2. one `Register` call in the modality's router registry;
3. its rates in `prices/vendor.json` (see [Prices](#prices)).

**The audit log holds configuration, not traffic.** `audit_log` keeps one row per change to
an agent config, skill, knowledge document, knowledge url, router config, plugin or policy:
the fields that moved with the value on each side, which client the change came from
(`dashboard`, `cli`, `sdk`, or `api` for a caller that named none) and who was at the
keyboard. Sessions, simulations, runs and calls are not in it — one agent produces
thousands a day, and they are read from their own lists. A write that moved nothing is not
recorded, bookkeeping columns are excluded from every diff, and a plugin's secret is never
in a row. Read it with `POST /v1/audit/query`, or ask what changed about one agent since
its directory was last synced with `GET /v1/agents/configs/{id}/changes`. Both are
server-side only.

Who made a change comes from three headers, read only from a server-side caller:
`X-Stream-Client`, `X-Stream-Actor-Id` and `X-Stream-Actor-Name`. Nothing signs them: they
buy a name beside a change somebody already held the credential to make, never permission
to make it.

### The request log

Everything above is what a request *did*. That it arrived at all is a log line, one per
request served, at info:

```
level=INFO msg="served a request" method=POST path=/v1/agents/sessions status=201 duration=425ms customer=examples bytes=877
level=INFO msg="served a request" method=GET path=/v1/agents/sessions/d84a…/events status=101 duration=94ms customer=examples
```

A socket is logged once, when it closes, so its `duration` is how long the connection lived
rather than a time to respond and its status is the `101` of the upgrade. A `5xx` is logged
at error instead, because an access log at a busy deployment is the one stream nobody reads
all of. A preflight is not logged, and the query string is never logged: a socket names its
customer there and a telephony vendor names a token.

### The usage service

`cmd/usage` is a second binary in the same image, with a database of its own. Producers ship
it usage events (`POST /v1/events`); it records each once by its `usage_id`, adds it to what
every subject the event names (its organization, app, key, user, and each category across
the organization and within the app) spent in the current daily, weekly, bi-weekly and
monthly window, and writes those totals to Redis before it answers, as
`usage:{<org>}:<subject>:<ids>` hashes with one `<interval>:<YYYYMMDD>` field per window.
Routers read the current window's field from Redis, and read through `POST /v1/windows` when
that field is missing, not only when the key is: a key written in an earlier window holds that
window's field alone. The ack's `mirrored` says whether Redis took a batch's totals; sending
the batch again rewrites them. It never prices: an event carries what the work cost. Nothing
ships to it yet.

```bash
USAGE_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/usage?sslmode=disable' \
USAGE_INGEST_TOKEN=local-usage-token USAGE_REDIS_URL=redis://localhost:56379 \
USAGE_MANAGE_KEY=local-usage-manage-key USAGE_TIERS_FILE=tiers/dev.json \
  go run ./cmd/usage            # or `docker compose build router && docker compose up usage`
go run ./cmd/usage --help       # every setting
```

**Accounts and tiers.** Every organization has an account: its credit (`granted_micros`,
the sum of its credit entries), its spend (`spent_micros`, debited in the transaction that
records its events, so a batch sent twice debits once), its status (`active` or
`suspended`), and its own value for any of its tier's settings (`limit_overrides`). A tier is
a string producers send on every event; the service assigns none. Its row holds its
`fail_mode` (what admission does while it cannot read a balance) and its `limits`: a spend
hard limit per UTC day and calendar month (0 or absent is none), per-channel counts,
recipients, countries and concurrency, sandbox bounds, and a low-balance threshold. A tier
string with no row gets the `default` row's, which cannot be deleted. The code names no tier
but `default`: the rows are data. At start the service adds the tiers of an operator's tier
file (`USAGE_TIERS_FILE`) that the database lacks, then a built-in `default` (fail closed,
nothing else) if there is still none; a tier the database has is changed only through the
API. `tiers/dev.json` holds made-up tiers for the local stack and the tests.

```bash
go run ./cmd/usage tiers check --file tiers/dev.json   # exits non-zero on the first error
go run ./cmd/usage tiers sync --dry-run                # what the next start would add
```

`POST /v1/standings` answers, per organization, the tier asked about (`fallback_tier` when it
has no row), the overrides and the settings in force, the status, the balance, and the
organization's spend in the current day and month. The management API under `/v1/manage`
(tiers, accounts, limits, credit, suspension, usage) takes `USAGE_MANAGE_KEY` in
`X-Manage-Key` and an actor in `X-Stream-Actor-Id`, never the ingest token; without the key
every route answers 404 `not_configured`. Each change is written to `manage_audit` with its
actor. A credit is written once per `Idempotency-Key`, and only its sign is checked. A
suspension, a restore and a change to an organization's settings need one too, so a request
sent again cannot make its change a second time: a replay with the same organization and body
changes nothing, writes no audit row and answers the standing as it is now. The key is one
space across the four writes: a key already used for another operation, organization or body
is a 409 `idempotency_key_reused`. Keys are kept in `manage_requests`, for good.

The service checks its own alerts, sends them to Sentry (`usage_alert:<name>`) and logs them.
A rejected event, a batch refused whole for what it holds (`batch_refused`: a 400, 413 or 422,
any 4xx but a 401), ingest failing and Redis refusing the totals page. A router's events
arriving late warns (other producers ship late by design and are not measured), and so does
a batch sent without the ingest token (`batch_unauthorized`, a 401): a misconfigured
producer, or no producer at all. `usage_batches_refused_total{status}` counts both kinds of
refusal.

## Prices

`prices/vendor.json` holds the vendors' published list prices, one entry per model (and per
service tier where a vendor charges one), embedded in the binary. It has no markup and no
negotiated figure. A deployment may lay a price file of its own over it with
`ROUTER_PRICES_FILE`: for one model and `when`, the first file with an entry wins, and its
`markup_bps` is the operator's markup.

At start, right after migrating, a router loads them into the shared database's
`price_changes` and `price_entries`: everything into an empty database, as one `seed` change
with the file's markup, and otherwise only the models the database has never priced, as one
`release` change. A model the database has priced keeps its rates, and the markup stays as
it is, whatever the files say now. Each load is audited (`price_list`, source `cli`). A load
that fails is retried every 30 seconds while the router serves; a file that does not parse
stops the router at start. Rows are only ever added: a trigger refuses to change or delete
an entry, and a change can only be cancelled before it takes effect. The one exception is
`router prices reload`, for a development database only, below.

| Command | What it does |
| --- | --- |
| `router prices show [--at <time>] [--json] [--unpriced]` | The list in force: each entry's vendor rates, its markup and what it charges; `--unpriced` lists the routed models no entry prices |
| `router prices sync --dry-run` | What the next start would load, writing nothing |
| `router prices check [--file <path>]` | Parses a price file (default `vendor.json`) and builds a list from it, with no database |
| `router prices drift [--since 168h] [--json]` | Per model, over every shard, how far each row's cost at the vendor's rates is from what the vendor said it charged (`attributes.vendor_cost_micros`) |
| `router prices export [--from <routing config>] [--phone] [--out <path>] [--strip]` | Writes a routing config's old `price:` blocks as a price file, and with `--strip` takes them out of it |
| `router prices reload --development [--dry-run]` | Development only: empties the price list and seeds it again from the price files, as a fresh database's first start does. See below |

A start never changes a rate the database already has, so on a laptop an edit to
`vendor.json` (a rate fixed, say) does not reach a database seeded before it.
`router prices reload --development` puts the files' list back: in one transaction, under the
lock a load takes, it empties `price_changes` and `price_entries` (every change goes, those
scheduled, cancelled or setting a markup included), then writes the `seed` a fresh database
gets from `ROUTER_PRICES_FILE` over `vendor.json`, with that file's markup. It first prints
the database it is pointed at, the address it connects to and the address the server answers
as (never the password), then what it removed, what it seeded, and the items whose price
moved; `--dry-run` prints the same and writes nothing. The audit log keeps its rows and gains
two: the removal (`deleted`, named `<time>_reload`), then the seed. Rows already recorded keep
the `price_version` that priced them, which then names no change. Running routers read the
new list within 30 seconds, so none needs a restart.

It refuses to run unless `--development` is passed and the database is on this machine or is
compose's: both the host `postgres.dsn` names and the address the driver connects to (a
`host` query parameter included) must be `localhost`, a loopback address, a unix socket or
`postgres` (compose's service, as the router's container reaches it). It also refuses under a
`ROUTER_ENV` other than unset, `local`, `development` or `testing`, but that is not what
protects a deployment: most leave `ROUTER_ENV` unset, which reads as `local`. Their remote
database host is what refuses it. A port-forward or proxy listening on localhost for a remote
database passes the host check, and only `--development` stands in its way, so read the
database and address the command prints first, and start with `--dry-run`.

`vendor.json` is built into the binary, so after editing it rebuild what runs the command: `go
run` does that itself, and the compose router needs `docker compose up -d --build router`
first, which also restarts it on the new image.

```bash
cd acceleration && go run ./cmd/router prices reload --development --dry-run
docker compose up -d --build router && docker compose exec router router prices reload --development
```

Until the router prices from the database, it prices every row from its price files:
`ROUTER_PRICES_FILE` over the embedded `vendor.json`, at the rates they give and with no
markup. A `price:` block in a routing config is ignored, with a warning at start;
`router prices export --from <your config> --out own-prices.json` moves one into a file for
`ROUTER_PRICES_FILE`, which keeps those models priced. The `check_prices` skill checks the list
against what each vendor charges.

## Transcripts, memory and phone

**Transcripts.** With `STREAM_API_KEY` and `STREAM_API_SECRET` set, `cmd/agent` writes every
relevant transcript committed by the flow controller and every reply into the Stream Chat
channel `agent:{agentID}`, off
the event stream rather than from inside the conversation loop. A voice call otherwise leaves
nothing behind, and any Stream Chat client can already read a channel.

For development, `ROUTER_CHAT_TIMINGS=true` (or `cmd/agent -chat-timings`) shows how fast each turn
was on the agent's reply, so somebody talking to the agent in a call UI sees it without reading
logs. Each reply gets a line after its text, such as
`⏱ reply 1112 ms = eou 452 + llm 412 + tts 248 · ttft 731 · ttfb 120`.
`reply` is the wait the caller felt, from the end of their speech to the first audible frame of the
reply (the first audio published, where the edge does not say when that was heard), and the three
stages add up to it: `eou` from the end of their speech until the turn was committed to
(transcription settling, cadence wait and end-of-turn decision), `llm` from then to the reply's
first text (absent when a reply started beside the decision was ready by then), and `tts` from the
first text to the first audible frame. The providers' own `ttft` and `ttfb` follow, because work
started early hides them inside the stages. A figure that did not happen is left out, and a turn
the caller talked over starts `⏱ interrupted ·`. The same figures, in whole milliseconds, are on
the message as the `timings` custom field (`reply_ms`, `eou_ms`, `llm_ms`, `tts_ms`, and the parts
and provider waits). Reading the conversation back, in a transcript or as the history of a bound
conversation, leaves the line out, so the agent never takes it for something said. Off by default.

**Memory.** With `MEM0_API_KEY` set, an agent recalls what it knows about the customer on
join and prepends it to its instructions, then hands each finished exchange over to be
learned from. Both happen off the conversation's path: recalling is bounded and a failure
means the agent starts the call knowing nothing rather than not taking it, and remembering is
queued and dropped under backpressure. `app_id` is the deployment and `user_id` is the
customer, so two deployments sharing one mem0 account do not read each other's memories.
Every call is recorded as a `requests` row with modality `memory`.

**Phone.** `phone.Provider` is the things vendors agree on: search, buy, release, point at
the bridge, dial out, press digits. All eleven vendors are declared in
[internal/phone/phone.yaml](internal/phone/phone.yaml). Eight are implemented; the other
three resolve to a stub that refuses every operation by name, so they list rather than being
absent.

Being implemented means different things for different vendors, so each declares its
`operations` and `GET /v1/phone/vendors` reports them. Seven of the eight can place an
outbound call. **DIDWW cannot, and never will here:** it sells numbers and has no call
control API at all, only SIP origination against a trunk resource it expects you to point
your own switch at. There is nothing to ask it to dial with, so `dial` is absent from its
operations and a number is not bought from it for an agent that has to call people.

Only Twilio and Telnyx also `attach`, which is the inbound direction: pointing a number at a
trunk so calling it reaches an agent. For the other five that is a per-vendor application
rather than a property of the number, and it is declared missing rather than half-done.

A customer who already has numbers and a SIP trunk at their own carrier can call out from
them without buying anything here. `POST /v1/phone/trunks` stores the trunk (host, port,
transport, username, codecs, and whether it accepts an INVITE without SDP); the password is
sealed under the deployment's key encryption key and never returned, and without such a key
the trunk paths answer 400 `not_configured`. `POST /v1/phone/trunks/{id}/numbers` records a number on it with
vendor `sip_trunk`. A call from that number is then placed by the router itself: it opens one
SIP dialog to the customer's trunk and one to the Stream trunk made for the call, and copies
SDP between them, so the audio goes from the carrier to Stream directly. `POST
/v1/phone/calls` answers `queued` at once, as with every vendor. A wrong password, host or
caller ID does not show in that answer. It shows only in the router's logs, under the call's
`vendor_call_id`. Behind NAT, the trunk must challenge the INVITE with 401 or 407: the router
corrects its Contact from the challenge, and without it the trunk's requests in the call,
such as its BYE, never reach the router. The node that placed a call
holds it until it ends and hangs it up if it stops, so a deploy ends calls in progress.
Initial digits, pressing digits and custom headers are refused for these numbers, and so is
`attach`: inbound calls to them reach the customer's own switch. `sip_trunk` is not in the
vendor registry, so it is never offered in `GET /v1/phone/vendors` or a number search.

Vendors also disagree about how a search can be narrowed. Telnyx filters by US state and
Sinch does not; Plivo matches digits only at the front of a number and Vonage matches at
either end but not both at once. So a provider declares which filters it can express, and
`GET /v1/phone/numbers/available` without a vendor asks every vendor that has its credentials
at once, merges what they offer cheapest first, and reports in `skipped` any vendor it could
not ask and why. Dropping a filter the vendor cannot express would answer a search for
Colorado with numbers from Ohio, which reads as a result. Capabilities are the exception:
every vendor says what its numbers carry even when it cannot filter on them, so those are
checked on the results.

Stream's SIP support is **inbound only** today. A number reaches an agent by the vendor
sending the call to a Stream inbound trunk; an outbound call is originated at the vendor and
bridged into the same trunk, because there is nothing to ask Stream to dial with. Attaching a
number creates the trunk and a routing rule whose caller id is a handlebars template, so the
SIP caller becomes a participant with a stable id that per-participant transcription can key
on.

### Placing a call

`POST /v1/phone/calls` makes its own trunk and routing rule and pins the rule to a call, so
the answered leg lands somewhere an agent can be waiting. The response says which call that
is, and an agent that is not in it hears nothing when the person picks up.

Vendors do not agree on what a call can be asked for, so a provider declares which terms its
API can express and a call naming one it cannot is refused rather than placed without it. A
ring timeout that was silently dropped is a call sitting in somebody's voicemail for a
minute.

| Term | Who can express it |
| --- | --- |
| `ring_timeout_seconds` | Everyone but Bird and Sinch |
| `initial_digits` | Twilio, Telnyx, Bandwidth, Sinch |
| `headers` | Nobody yet: Telnyx's TeXML call is not known to put them on the SIP leg |

Getting past the trunk splits the vendors in two. Twilio, Telnyx and Plivo can name SIP
digest credentials in their call plans, so the trunk's own password is enough. Vonage, Bird,
Sinch and Bandwidth have no field for a password anywhere, so their trunk recognises them by
the address they call from, declared as `signalling` in `phone.yaml`. Stream reads an empty
allowlist as "accept everything" rather than "password only", so a vendor that authenticates
by address refuses to place a call until its addresses are declared: a trunk with neither is
a way into a customer's calls for anyone who learns its uri. Vonage's and Bird's addresses
are published and are in `phone.yaml`; Bandwidth's and Sinch's come with the account, so
those are per deployment.

Plivo and Bandwidth also will not take a call plan on the request at all, and fetch one when
the person answers. For those, placing a call parks the plan and hands the vendor a
single-use expiring token at `GET /v1/phone/answer/{token}`, which is the one unauthenticated
path here because the vendor fetching it has no customer to name. It needs `ROUTER_PUBLIC_URL`
to be set to somewhere those vendors can reach, and says so rather than placing a call
nothing will answer. Sinch is the near miss: its callbacks belong to the application rather
than to a call, so its plan travels inline on the callout instead.

```bash
go run ./cmd/phone dial -from +15125551234 -to +15550001111 \
  -call-id support-line -ring-timeout 20s -digits ww1234# -custom reason=callback
```

Buying a number spends money every month, so nothing automated buys one. The runbook, run by
hand:

```bash
# What is on offer, and what it costs
go run ./cmd/phone search -vendor telnyx -country US -state CO \
  -type local -feature hd_voice -feature emergency -limit 5

# Buy one of them. -country is only needed by the vendors that sell out of a
# country's inventory rather than by number, which the search reports.
go run ./cmd/phone buy -vendor telnyx -number +1719XXXXXXX -tag project=sprint12
go run ./cmd/phone list

# Give it back, which is what stops the charge
go run ./cmd/phone release -number +1719XXXXXXX
```

`cmd/router` applies migrations on startup. To run them by hand:

```bash
goose -dir migrations postgres "$ROUTER_POSTGRES_DSN" up
```

## Moving a customer

`api_key` mode cannot bootstrap itself over HTTP: every endpoint wants a key and there is
nobody to issue the first one. `keys create` does it beside the database, printing the
secret once and never logging it:

```bash
go run ./cmd/router keys create --org Acme --app-name "Acme production" --key-name default
```

`--app-id` keeps an id the customer already has, which is what makes an import land under
something. Running it again for an existing app adds a second key rather than refusing.

Three endpoints move a customer between two deployments, all server-side only:

| Endpoint | What it does |
| --- | --- |
| `GET /v1/data/export` | Everything the calling app has, one JSON object per line, ending with the cursor its changes carry on from |
| `POST /v1/data/import` | Writes an export, or a batch of changes, into this deployment |
| `GET /v1/data/changes?after=` | What has happened to the app's rows since that cursor, and whether that is all of it |

`cmd/router replicate` is both halves at once: the copy, and everything that happens after
the copy, which is what makes the switchover free of a window where writes are lost.

```bash
go run ./cmd/router replicate --from https://old.example.com \
  --api-key vak_live_… --api-secret vas_live_… [--as <app id>] [--once]
```

The customer is always the authenticated caller, never anything in the body or the query,
so an export is the caller's own data and an import lands under the caller's own app
however the file was edited on the way. An import upserts and refuses a row whose primary
key belongs to another tenant, so replaying the same change twice lands where applying it
once would and a guessed id overwrites nothing. A request row is the exception: each
deployment numbers its own, so it is imported by its `usage_id` with an id the destination
gives it, written once, and a delete of it is replayed by its `usage_id`, or for a row from
before usage ids by the one its import made from its source id.

Key rows, sealed secrets, OAuth access and refresh tokens, organizations and apps are not
exported. A moved deployment mints its own keys and re-authorizes its plugin connections.
Voice recordings live in the bucket rather than in Postgres, so their rows move and copying
the bucket is the operator's.

Changes are captured by a trigger per table writing into `data_changes`, gated on a row in
`data_change_capture`: with nobody moving, the hot path costs one index probe. Exporting
starts the capture and reading a page refreshes it, so an abandoned move stops recording by
itself after `ROUTER_DATA_MOVE_RETENTION`, and a cursor older than that is answered 410
rather than silently missing writes. Each change records `pg_current_xact_id()`, and the
cursor never passes `pg_snapshot_xmin(pg_current_snapshot())`, so a sequence number handed
out by a transaction that has not committed yet cannot be stepped over.

## Test

```bash
# Unit tests, no credentials or services needed
go test ./...

# Integration tests hit real providers, Postgres and Redis
go test -tags integration ./...

# Manual live full-duplex proof for GPT-Live (requires OPENAI_API_KEY and ffmpeg)
go test -tags integration -count=1 -v -run '^TestOpenAILiveIntegrationSuite/TestTheModelHearsAndAnswersWhileTheCallerIsStillSpeaking$' ./internal/sts/openailive
```

Integration tests skip themselves when the credentials or services they need are absent.
CI checks caller-audio forwarding against a fake Live socket and compiles the shared STS
integration suite without calling providers. The full-duplex proof is manual; providers that
do not declare `FullDuplex` or the needed transcript capabilities skip it.

`internal/agent/conversation_test.go` is the conversation-quality suite: a synthesised caller
is played into a real agent at the rate a real call delivers audio, in silence, in a noisy
room, over somebody else's conversation and while the agent is talking. What it asserts is
what a caller would notice, including how long they waited to be answered.

```bash
go test -tags integration -run TestConversationSuite ./internal/agent
```

Postgres and Redis for local runs:

```bash
docker run -d --name va-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=model_router \
  -p 55432:5432 postgres:16-alpine
docker run -d --name va-redis -p 56379:6379 redis:7-alpine
```

The API suites run as one database by default. To run them as a shard does, against a
second database and Redis:

```bash
docker run -d --name va-redis-shard -p 56380:6379 redis:7-alpine
ROUTER_POSTGRES_SHARD_DSN='postgres://postgres:postgres@localhost:55432/model_router_shard?sslmode=disable' \
ROUTER_REDIS_SHARD_ADDR=localhost:56380 go test -tags integration ./internal/api
```

`internal/store` runs its suite both ways in every run.

## Regenerate the HTTP layer

Operations are declared in Go with Huma, on a chi router, and the structs are the source
of truth. `api/openapi.yaml` is rendered from them by `cmd/openapi` and read by every client
generator, so it is never edited by hand. After changing an operation:

```bash
go run ./cmd/openapi
uv run ../plugins/stream/generate.py
uv run ../sdks/swift/generate.py
(cd ../sdks/js && npm run types)
(cd ../dashboard && npm run types)
```

The usage service's operations are a spec of their own, `internal/usage/server/openapi.yaml`,
which no SDK is generated from. Render it from the service's binary, whose error responses it
describes: `go run ./cmd/usage openapi`.

The Swift client is generated from a filter rather than from the whole spec, and
`generate.py --check` fails if anything in that filter is not marked `x-client-accessible`.
Dropping that mark is therefore the whole of removing an operation from the iOS SDK.

The JavaScript client generates types and no runtime, because one package has to work in a
browser and on a server and the runtime is the part that differs between them. Its request
code is hand-written in `sdks/js/src/client.ts` and typed against the generated `paths`, so
a new operation is reachable there as soon as the types are regenerated.

The three sockets cannot be Huma operations, since an upgrade returns a connection rather than
a response, and neither can the log stream, the data export and import, which stream, or the
plugin callback and the connector consent's launch page, handoff and callback, which a browser
reaches. Their handlers are written by hand in `internal/api/sessionws.go`, `streamws.go`,
`dispatchws.go`, `logs.go`, `datamove.go`, `plugins.go` and `authorizations.go`, and the Python side of the sockets in `plugins/stream/.../_socket.py`. Each is
still declared, with its `101` or streamed response, in `internal/api/handwritten.go`, so a
reader and a client generator know it exists and the server-side check reads its marks like
any other operation's: an operation the spec cannot see is the one place a default that
refuses could fail open.

What one node asks another is not part of this API and is not in the spec. It is the
`Node` gRPC service in `internal/node/node.proto`, generated by buf. Both buf and the
protobuf plugins are pinned `go run` commands rather than tool directives, so nothing has
to be installed and a code generator's dependencies stay out of the router's:

```bash
go run github.com/bufbuild/buf/cmd/buf@v1.73.0 generate
```

## Design notes

- **The core knows nothing about modalities.** `routing.Router[P]` resolves a target, ranks
  candidates and fails over; each modality adds only a provider contract and a session that
  knows which of its events count as a unit of work.
- **One row per unit of work.** A completed turn for speech-to-text, a completed synthesis
  for text-to-speech, a completed completion for an LLM, one question asked for a search,
  alongside rows for sessions that
  failed to start. Everything is keyed by `modality`, so the same provider can serve two of
  them without its numbers mixing.
- **Latency is the number the customer felt.** For speech-to-text that is the provider's
  decode time; for text-to-speech it is the wait for the first audio, since the rest
  arrives while the listener is already hearing it; for an LLM it is the wait for the first
  token, for the same reason.
- **A conversation lives in the caller, not the provider.** Every LLM request carries the
  whole history, so consecutive turns can be served by different providers and a failover
  loses nothing.
- **Delegation is not tool calling.** Standardising a tool schema across providers is a
  bigger question than routing needs answered, and a tool call is the wrong shape for a
  phone call anyway: it blocks the reply on the result. The harness has the fast model ask
  a slower one for help in its own words, in the middle of a sentence it goes on to finish.
- **Costs come from the price list, not from the provider.** The recorder prices every row
  once, through one price list (`internal/pricing`) of the vendors' rates in
  `prices/vendor.json` (see [Prices](#prices)). A model with no price reports a cost of zero
  rather than a guess, and is logged as billable work with no price unless it is marked
  `free` or `local`.
- **Transcript modes.** `delta` appends, `replacement` supersedes the current in-progress
  text, `final` is settled. Both speech-to-text providers send replacements.
- **An audio chunk does not say whether it is the last.** A streaming voice only learns
  that after the fact, and buffering a chunk to find out would cost the latency the design
  is for, so `SynthesisComplete` is what ends an utterance.
- **A speech-to-speech model owns its own turns.** It is the voice activity detector, the
  turn detector, the transcriber and the voice at once, so nothing in front of it acts as
  one. `SpeechStarted` says only that the model heard someone; `ResponseComplete` with
  `Interrupted` is the one signal a consumer drops its buffer on, and the router session
  drops any audio of that reply still arriving, because the model learns of a barge-in a
  round trip after the caller. What a model cannot do is refused, never dropped: a typed
  turn to a model that takes none, or instructions to one that took them at setup, comes
  back as an error.
- **Uptime and latency come from the same request rows** as billing, so there is no
  separate health-probe pipeline to keep in sync.
- **Nothing a conversation does waits on a database.** Turns, transcripts and memories all go
  through a bounded queue and are dropped when the writer falls behind, because losing a row
  costs a statistic while blocking costs the participant a silence.
- **Failover happens at session start.** A candidate that fails to start is recorded and
  the next one is tried; a provider that keeps failing falls down the ranking.
- **`customer_id` is a trusted header.** Real authentication is not part of this version.

## Source research is the caller's, not the backend's

The backend holds no research workspaces. An agent that reads source owns its own
Daytona VM and offers `investigate_sdk` as one of its own tools, which the session
dispatches back to it like any other caller-owned tool. Nothing about the VM, the
repositories it holds or the credentials it reads them with is the backend's to know.

This is deliberate. The agent has to be told which repositories it may read and under
whose Cursor and Daytona keys, and every one of those answers belongs to whoever
deploys the agent rather than to whoever runs this router. A profile on an agent config
could only ever be a copy of a decision made elsewhere.

`agents.Daytona()` / `sandbox: daytona` is unaffected: that is delegated Python
execution for code the subagent writes, and remains the backend's to run.

See `../../artemis-impl` in the sibling workspace for an agent that does own one: the
runnable support application, its repository catalog and the explicit docs compiler
contract.

## Persistent text conversations

Every text session is kept in a Stream Chat `agent:support-<uuid>` channel unless it is
`incognito`; pass `conversation_id` to bind to an existing one. The configured agent ID is
preserved; the backend verifies the channel's customer and agent ownership on
resume and on `GET /v1/agents/conversations/{cid}/messages?agent_id=...&before=...`.
This endpoint returns up to 100 messages and a `before` cursor. Completed ordinary
user/assistant messages restore model context, capped at 100 messages and 60,000
characters. `context_truncated` reports the bound. Stored tool attachments are
never executed or restored as system instructions.

`conversation_updated` WebSocket events carry a cumulative visible message snapshot:
`id`, `question_id`, `state`, `response_started_at`, `state_started_at`, final
`finished_at`/`duration_ms`, text, attachments and `saved`/`persistence_error`.
States are `thinking`, `queued`, `tools`, `writing`, `completed`, `failed`,
`cancelled`, or `interrupted`. Thinking conveys activity, not private reasoning.
Each `tool_calling` attachment has an immutable backend `started_at`, stable
`tool_call_id`, name/title, status/phase, structured summary, optional
`execution_started_at`, and final `finished_at`/`duration_ms`. Raw tool arguments
and results are excluded. Internal model turns remain bound to the original
question so a late interruption cannot cancel its successor.

The backend owns all Chat writes. Initial messages, tool starts/completions and
final results use durable writes; intermediate cumulative snapshots use
`EphemeralMessageUpdate`, throttled to 200 ms when changed. Nothing is written to
local disk: a held conversation retries its pending writes in memory with stable
message IDs, and Stream Chat is the durable copy. Pending data stays visibly
unsaved and is overlaid on resumed history. A conversation nobody holds is rebuilt
from its channel: a user message is stored under its command ID, so a command in
the last 100 messages is never run twice, and a reply still running there when it
is reopened is marked interrupted. Writes not yet in Stream when the process stops
are lost. Stream credentials are required; Postgres and Redis are not. Closing the
last client of a persisted text session ends that session; the saved channel
remains available.

The Go SDK exposes `agents.ChatOptions{ConversationID: cid}`,
`Session.ConversationID()`, `Session.ContextTruncated()`,
`stream.Backend.ConversationHistory`, and `Event.ConversationMessage`.
Without Stream credentials a new text session runs without a channel. Voice calls write
their transcript into Chat as well: what a caller is saying streams as ephemeral updates
and is stored once the turn settles. Text-mode native skills
use task-oriented delegation; voice sessions retain their speech-repair policy.

STT/TTS recording requests may explicitly set `inline: true` to complete a short
request synchronously without a database. The usual 202 response contains a
completed/failed result, not a pollable job. Audio is not stored. Inline requests
reject callbacks, transcription URLs, audio over 8 MiB and text over 16,000
characters; the request's cancellation propagates and execution is bounded to
90 seconds. Existing asynchronous recording behavior is unchanged. In Go use
`Recorded{Audio: clip, Inline: true}` or `TTS().InlineRecording(...)`.
