# AGENTS.md

When running an agent using @acceleration/ or an example using @acceleration/ refer to the skill acceleration/development_skills/dashboard_skill.md to setup the dashboard and launch the example. Make sure you restart any routers or binaries to avoid running old builds.

Work happens on the `accelerate` branch here, and the agents docs live in getstream.io's `content/docs` (see the `docs` skill).

## What is here

| Path              | What it is                                                              |
| ----------------- | ----------------------------------------------------------------------- |
| `acceleration/`   | Go model router: STT, TTS, LLM and search behind one API, plus the agent that joins a call. `cmd/router` serves it |
| `dashboard/`      | Next.js app for watching calls and editing agent configs. Talks to the router from the browser |
| `sdks/python/`    | The Python SDK: `Agent`, `Runner`, the plugin contracts                  |
| `sdks/js/`        | The JavaScript SDK, one package for Node and the browser: every endpoint typed from the spec, sessions, tools, agent dispatch |
| `plugins/`        | 44 Python packages, one per provider. `plugins/stream` is the client for `acceleration/` |
| `sdks/swift/`     | Three iOS packages: `core` (state and API), `ui` (SwiftUI), `rtc` (voice over Stream Video) |
| `examples/voice_agents/` | Runnable agents. `simple_voice_ai` is the smallest one                  |
| `tui/`            | Go terminal UI for a conversation with an agent: the terminal counterpart of `dashboard/` |
| `sdks/kotlin/`, `sdks/dart/` | Client-side SDKs for Android and Flutter, each a core, ui and rtc package like Swift's |
| `sdks/dotnet/`, `sdks/ruby/`, `sdks/rust/`, `sdks/php/` | Server-side SDKs: agents, dispatch, folder sync and the router. Built and tested in Docker; each has an `sdk-<lang>` skill |
| `sdks/go/`, `benchmark/` | The Go SDK and the voice benchmark. `go.work` ties the Go modules together |

## SDK changes

When you need to make an SDK change start by only changing Go
Leave a note in the sdk skill at the bottom that other SDKs also need to udpate.
We run this periodically, you don't need to change all SDKs at once.
Go goes first, and later on we update the others.

## Local dev

Everything reads the repo-root `.env` for provider credentials.

Run the router in Docker, with the Volt dashboard's override (see the `dashboard` skill):

```bash
docker compose -f compose.yaml -f ../volt-dashboard/docs/local-agents/compose.volt.yaml up -d --build router
```

That serves the router on `:8080`, with Postgres on `:55432` and Redis on `:56379`, its data
in the `vision-agents_pgdata` volume. Rerun it after router changes so you never test an old
build. Those two ports are also what the standalone `va-pg` and `va-redis` containers use, so
stop those first if they are running.

`.env` must set `ROUTER_AUTH_KEK` (single-quoted: compose expands a `$` in it) and
`ROUTER_PUBLIC_URL=http://localhost:8080`. Without the KEK no plugin client secret can be
saved, and changing it leaves the stored ones unreadable.

Logs: `docker compose logs -f router`.

An agent, once the router is up:

```bash
cd examples/voice_agents/simple_voice_ai
uv sync && uv run simple_voice_ai.py run
```

It prints a call URL on the dashboard; open it and talk.

The client generators (`plugins/stream/generate.py`, `sdks/swift/generate.py`) are standalone uv scripts with inline metadata: `uv run <script>` does not resolve the workspace.

Python commands all use `uv`. Never `python -m`. If you hit dependency issues, stop and ask.

```bash
uv run --no-sync dev.py check              # ruff + mypy + unit tests
uv run --no-sync pytest -m "not integration"
uv run --no-sync pytest -m "integration"   # needs .env secrets
uv run --no-sync ruff check .
uv run --no-sync ruff format .
uv run --no-sync mypy
```

`--no-sync` avoids a uv panic in sandboxed environments.

## HTTP API

The router serves its API with [chi](https://github.com/go-chi/chi) and [Huma](https://huma.rocks).
The Go structs are the source of truth: `acceleration/api/openapi.yaml` is rendered from them and
is never edited by hand.

- Declare an operation with `huma.Register` in the file for its resource, with its request and
  response bodies as Go structs beside it. `internal/api/policies.go` is the example to copy.
- Describe fields with `doc:` tags and constrain them with `minimum:`, `enum:`, `readOnly:` and
  the rest, so what validates a request is also what documents it. A type's own description goes
  in a `TransformSchema` method, and a named string enum in a `Schema` method using `namedEnum`.
- Fail with an `APIError` (`internal/api/apierror.go`): `invalidRequest(...)`, `notFound(...)`
  and their siblings, whose type decides the status. A failure answered from several places is
  an `APIError` value of its own, named `errX` (`errUnknownConfig`, `errNoSessions`, built with
  `notConfigured(...)` for a feature the deployment lacks), returned as is: one code, one status,
  everywhere. Every failure is the `{"error": {"message", "type", "code",
  "doc_url"}}` envelope; any other error an operation returns is a 500 saying only "something went
  wrong", logged with its stack. A request that fails validation is a 400 `validation_failed`.
- An operation is server-side only unless it sets `Extensions: {"x-client-accessible": true}`.
- List endpoints page by cursor. Read the `pagination` skill
  (`.claude/skills/pagination/SKILL.md`) before adding one or a `limit` parameter.
- After changing an operation, run `go run ./cmd/openapi` in `acceleration/`, then regenerate the
  clients (see [acceleration/README.md](acceleration/README.md)). A test fails if the committed
  `openapi.yaml` is out of date.

- A query parameter a request may leave out is an `optionalParam[T]`; `ptr()` is nil when it was
  left out. Huma takes no pointer for one.
- A route served by hand (a socket, a stream) is still declared, in `internal/api/handwritten.go`,
  so readers and client generators see it and the server-side check reads its marks.

There is no hand-written spec any more: every operation, socket included, is declared in Go.

The JavaScript SDK is its own npm package, checked with node 22 and no runtime dependencies:

```bash
cd sdks/js
npm install
npm run types   # regenerate src/generated/api.ts from the spec; --check in CI
npm test        # typecheck, then the suite against a real http and ws server
```

## Benchmark

To run Voicebench (`benchmark/`) locally or read why a run failed, read the `voicebench` skill
(`.claude/skills/voicebench/SKILL.md`) first: the keys and services a run needs, which command
answers which question, and how to read the report.

## Testing

For Go tests, read the `go-testing` skill (`.claude/skills/go-testing/SKILL.md`) first: testify
suites, the shared `RouterSuite` for integration tests, and how to wait for async writes.

Before you run more than one coding agent at once (parallel PRs, reviewers, fixers), read the
`parallel-agents` skill (`.claude/skills/parallel-agents/SKILL.md`): test databases, migration
slots, merge order and how not to burn tokens.

- Framework: pytest. Never mock.
- `@pytest.mark.asyncio` is not needed (asyncio_mode = auto).
- Integration tests use `@pytest.mark.integration`.
- NEVER adjust `sys.path`.
- Keep unit-tests for the class under the same test class. Do not spread them around different test classes. For example, tests for `Agent` must be inside `TestAgent`, etc.
- ALWAYS test behavior, not calling a path.
- Use pytest.fixture for test setup, not helper methods
- NEVER observe method calls in tests; assert on outputs and state.

## Python rules

- Never use `from __future__ import annotations`.
- Prefer specific exceptions if they are known. If the exception type is not clear, it is ok to use `except Exception as e`.
- Avoid `getattr`, `hasattr`, `delattr`, `setattr`; prefer normal attribute access.
- Docstrings: Google style, keep them short.
- Do not use section comments like `# -- some section --`
- Prefer `logger.exception()` when logging an error with a traceback instead of `logger.error("Error: {exc}")`
- Do not use local imports, import at the top of the module
- Avoid `# type: ignore` comments.
- Avoid using `Any` type.
- When adding code to an existing file, follow the patterns already established in that file (e.g. error handling style, import guards, naming).

## Code style

### Imports:

- ordered as: stdlib, third-party, local package, relative. Use `TYPE_CHECKING` guard for imports only needed by type annotations.
- Never import from private modules (`_foo`) outside of the package's own `__init__.py`. Use the public re-export (e.g. `from vision_agents.testing import TestResponse`, not
  `from vision_agents.testing._run_result import TestResponse`).

### Naming:

- private attributes and methods use a leading underscore (`_sessions`, `_warmup_agent`). Public API is plain snake_case.

### Type annotations:

- use them everywhere. Modern syntax: `X | Y` unions, `dict[str, T]` generics, full `Callable` signatures, `Optional` for nullable params.

### Logging:

module-level `logger = logging.getLogger(__name__)`. Use `debug` for lifecycle, `info` for notable events, `error` for failures without a traceback,
`exception` for errors with traceback.

- In hot paths (audio processing, event handling), guard debug logging behind `if logger.isEnabledFor(logging.DEBUG):` to avoid formatting overhead when debug is disabled.

### Constructor validation:

- raise `ValueError` with a descriptive message for invalid args. Prefer custom domain exceptions over generic ones.

### Async patterns:

- async-first lifecycle methods (`start`/`stop`). Support `__aenter__`/`__aexit__` for context manager usage.
- Use `asyncio.Lock`, `asyncio.Task`, `asyncio.gather` for concurrency.
- Clean up resources in `finally` blocks.

### Method order:

- `__init__`, public lifecycle methods, properties, public feature methods, private helpers, dunder methods.

### Other

- Smallest possible diff. Prefer deleting code over adding it.
- Don't add error handling, logging, validation, comments, abstractions, config options, or "future-proofing" I didn't
  ask for.
- Match the style and abstraction level of surrounding code. Don't introduce new patterns or helpers unless asked.
- Fix root causes, not symptoms. No try/except to swallow bugs.
- Change only what I asked for. Don't refactor adjacent code — ask first.
- Do not remove valid comments when editing/refactoring code.

## Plugins

- In every `plugins/*/pyproject.toml`, the wheel target must be `packages = ["vision_agents"]`. Listing `"."` pulls `tests/`, `README.md`, `example/`, etc. into the published wheel.
- Each plugin must keep `readme = "README.md"` in `[project]` and a `README.md` next to its `pyproject.toml` so PyPI renders a description page.

## Token efficiency

- When making multiple related changes to the same file, combine them into fewer Edit calls with enough surrounding context, rather than one edit per change.
- Run tests with Bash directly. Only use subagents for test runs when you need to do other work in parallel.
- Only use TodoWrite for tasks with 5+ steps. Don't update it after every individual edit.

## Changelog

- Lives in `CHANGELOG.md` at the repo root.
- Organised by version heading (`# v0.4.0`), then sections: **Breaking Changes**, **New Features**, **Bug Fixes**.
- Only include user-facing changes (public API breaks, features, fixes). Skip docs-only and CI-only commits.
- Reference PR numbers inline, e.g. `(#374)`.
- To generate: `git log <last-tag>..HEAD --oneline --no-merges`, then classify each commit.
