---
name: sdk-php
description: How to build and extend the PHP SDK in sdks/php. Read this before changing the client, the generated DTOs, the Amp worker (dispatch, watch, realtime), folder sync, or before reaching for a dependency.
---

# PHP SDK conventions

The per-language half of [sdk](../sdk/SKILL.md), for the server side only. It records the
decisions [`sdks/php`](../../../sdks/php) already follows, so a change lands consistently
rather than re-litigating them.

One Composer package, `getstream/vision-agents`, namespace `GetStream\VisionAgents`. PHP ^8.4,
tested on 8.4 and 8.5. Strict types everywhere, `final readonly` classes, backed enums.

## Layout

```
src/
  Client.php Backend.php Http.php Json.php     core: PSR-18 transport, auth, tolerant JSON
  Agent.php Session.php Sessions.php Responses.php Items.php AgentResponse.php AgentHandle.php
  Folder.php Folder/ Skill.php Harness.php Pipeline.php Tools.php Knowledge.php
  Edge.php Edge/StreamTransport.php Call.php   the Stream call, through getstream/getstream-php
  Router.php Router/                           search, recordings, realtime entry points
  Inbound/                                     InboundCall, InboundMessage
  Worker/                                      Amp only: Socket, Dispatch, Watch, Realtime
  Exception/
  Generated/                                   bin/generate.php output, never edited
bin/generate.php
tests/Unit tests/Live tests/Support tests/server/router.php
```

## Why these choices

- **Name.** `getstream/vision-agents` sits beside `getstream/getstream-php` on Packagist.
  Packagist needs `composer.json` at a repository root, so publishing is a splitsh/lite
  subtree split of `sdks/php` into its own repo from CI. Do not publish from the monorepo.
- **PHP 8.4.** 8.3 left active support at the end of 2025; 8.4 gives `new Foo()->bar()`, property
  hooks if ever wanted, and is what the official images ship. Readonly classes and typed
  class constants need 8.3+ anyway.
- **PSR-18/17 plus `php-http/discovery`.** A PHP app already owns an HTTP client (Guzzle,
  Symfony). Exposing Guzzle would force a version on them. Discovery finds whatever is
  installed; the constructor takes any client and factories. Discovery's Composer plugin is
  disabled (`allow-plugins: false`): it must never install packages behind a user's back.
- **Generation: a small in-repo generator, DTOs only.** openapi-generator's `php-nextgen` is
  still beta and emits a Guzzle-bound client per operation; Jane is Symfony-shaped; Stainless
  is a hosted product. The spec is the router's own and uses a narrow subset of OpenAPI, so
  `bin/generate.php` (about 300 lines, Symfony Yaml) writes one `final readonly` class per
  object schema and one backed enum per enum schema, deterministic and diffable, with a
  `--check` mode for CI. Operations are not generated: `Client` has one method per HTTP verb,
  as in JS, and the hand-written resources hydrate the DTOs.
- **The core/worker split.** A PHP web request is short-lived; a websocket is not. So the core
  (everything outside `Worker/`) is synchronous PSR-18 and has no event-loop dependency: it
  runs under FPM, Laravel, Symfony. Anything holding a socket (`Session::watch`, `Dispatch`,
  `Router::*->realtime()`) lives in `Worker/` and needs `amphp/websocket-client` ^2 (Revolt,
  fibers), listed under `suggest`, never `require`. Amp over ReactPHP because it is
  fiber-based: handlers are plain blocking-looking code, and the same `Client` works inside
  and outside the loop.
- **Amp's PSR-18 adapter is preferred when installed** (`Http::client()`), so an HTTP request
  made from a dispatch handler suspends its fiber instead of stalling every other call the
  worker holds. It is built with `DnsSocketConnector` (Amp's default connector retries a
  refused connection with backoff: a dead router took 18 s to fail) and a 120 s transfer
  timeout (opening a session returns only once the agent is in the call).
- **Stream's PHP SDK for calls.** `Edge` uses `getstream/getstream-php` ^12.1 (`VideoClient`),
  sent through `Edge\StreamTransport`, an adapter over our PSR-18 client, so it too suspends
  in a worker, and tests point it at the local server. It needs a secret of 32+ bytes
  (firebase/php-jwt refuses shorter). Built lazily: an agent that only chats needs no Stream
  credentials.

## Rules

- **Null means omit.** Every optional DTO field is nullable and `toArray()` drops nulls. Never
  copy a schema default into PHP: the router or the stored config decides.
- **Tolerant reading.** `Json::*` readers never throw on a missing or mistyped field. Enums
  are `Enum|string`: an unknown value is kept as the raw string, never refused. Date-times are
  `DateTimeImmutable`, fractional seconds cut to 6 digits (Go sends 9).
- **Errors.** `RouterException` carries `status` (0 when nothing arrived), `operation`
  (`METHOD /path`), `said` (the router's `error`), `body`, `retryAfter`. Local misuse is
  `ConfigurationException`, raised before a request. Socket `error` frames are
  `RealtimeException`; a failed recording job is `RecordingFailedException`.
- **Auth is `Backend::headers()`**, minted per request. Server sockets send the same headers
  on the handshake, never credentials in the query string.
- **No reconnect.** A dropped socket ends the iterator or `run()`; the owner decides. A frame
  that is not a JSON object, or of an unknown type, is skipped, never fatal.
- **Tool results repeat `command_id` and `turn_id`** from the `tool_call` when present, or a
  durable command's result is refused. Tools run in their own fiber; a `tool_cancel` drops
  the answer (a PHP callable cannot be interrupted).
- **Dispatch** answers `accepted`/`rejected` for calls only, reports `load {active_agents,
  latency_ms}` after a `ping`/`pong` round trip, drains running handlers on exit, and traps
  SIGINT/SIGTERM only when pcntl is loaded (the official image has none).
  `getOrCreateAgent` holds a per-channel `LocalKeyedMutex` so two messages cannot start two
  agents.
- **Folder sync ports Go exactly.** `Folder::fingerprint` is Go's `fingerprint`; with only a
  subagent and cost labels set, `Agent::sync` produces Go's hash (`fmt.Sprint` of a map is
  `map[k:v ...]`, keys sorted). The fixture hash `02a7b2c8428f31e3a2b93ca2f5a6ec70` is pinned
  in `FolderTest`; if it moves, the port is wrong, not the pin. `.agent_sync` is
  `{"hash","synced_at":"...+00:00"}`; a matching stamp reads the stored config instead of
  syncing. Unknown `agent.yaml` keys are refused.
- **Rewind takes a response `id`**, never a `turn_id`, and answers 204. Fork takes
  `responseId` in `ForkSessionRequest`. A persisted conversation refuses rewind with 400.
- **`responses->create()` returns while the response is `running`.** It does not wait; the
  live tests poll `responses->list()`.
- **Realtime `start` frames nest options under the modality key** (`{"type":"start",
  "config_id":..., "stt":{...}}`). The router reads only its own modality's block.
- Comments state constraints only. Docblocks are short.

## Testing

- **No mocks.** Unit tests run against `php -S` executing `tests/server/router.php`, which
  answers from a JSON script and records every request; assertions are on what was sent.
  Socket tests run a real `amphp/websocket-server` on the same Revolt loop
  (`tests/Support/LocalSocketServer`), playing the router's side.
- Live tests (`tests/Live`) are skipped unless `VISION_AGENTS_URL` is set;
  `VISION_AGENTS_CUSTOMER_ID` defaults to `examples`. Never start or restart the router for
  them.
- PHPStan at `level: max` with `phpstan-strict-rules`, over `src`, `tests` and `bin`
  (`tests/server` excluded: it is a script for `php -S`).

## Docker

The host has no PHP. Everything runs in the official images; nothing is installed on the host.

```bash
docker run --rm -v "$PWD":/repo -v "$PWD/sdks/php/build/composer-cache":/tmp/composer-cache \
  -e COMPOSER_CACHE_DIR=/tmp/composer-cache -w /repo/sdks/php composer:2 install
docker run --rm -v "$PWD":/repo -w /repo/sdks/php php:8.4-cli vendor/bin/phpunit --testsuite unit
docker run --rm -v "$PWD":/repo -w /repo/sdks/php php:8.4-cli vendor/bin/phpstan analyse --memory-limit=1G
docker run --rm -v "$PWD":/repo -w /repo/sdks/php php:8.4-cli php bin/generate.php ../../acceleration/api/openapi.yaml --check
docker run --rm -v "$PWD":/repo -w /repo/sdks/php -e VISION_AGENTS_URL=http://host.docker.internal:8091 \
  php:8.4-cli vendor/bin/phpunit --testsuite live
```

Mount the repo root, not `sdks/php`: the generator reads `../../acceleration/api/openapi.yaml`.
`composer:2` runs Composer (it has zip); `php:8.4-cli` and `php:8.5-cli` run the tests. The
Composer cache lives in `build/`, which is gitignored. A custom Dockerfile was dropped when the
Docker VM disk was full; do not prune shared images or volumes to make room.

## After the spec changes

1. `php bin/generate.php ../../acceleration/api/openapi.yaml` and review the diff.
2. Hand-written resources only change for a new behaviour, not a new field.
3. Unit tests, PHPStan, then the live suite once.
4. Copy `.claude/skills/sdk/SKILL.md` and `acceleration/api/openapi.yaml` byte for byte into
   `sdks/php/.sdk_update_log/`, so the next update can diff what changed since.

## Review checklist

- A new optional field defaulted in PHP instead of left null.
- An enum read with `from()` instead of tolerating unknown values.
- A socket, event loop or Amp class referenced outside `Worker/` (except the guarded
  `Session::watch` and `Router` entry points, which only construct worker objects).
- A blocking `sleep`/`usleep` in code a worker calls (use `Pause::for`).
- A mock, or an assertion on which method ran instead of what went over the wire.
- `agent.yaml` handling that silently ignores a key.
