# AGENTS.md

## What is here

| Path              | What it is                                                              |
| ----------------- | ----------------------------------------------------------------------- |
| `acceleration/`   | Go model router: STT, TTS, LLM and search behind one API, plus the agent that joins a call. `cmd/router` serves it |
| `dashboard/`      | Next.js app for watching calls and editing agent configs. Talks to the router from the browser |
| `sdks/python/`    | The Python SDK: `Agent`, `Runner`, the plugin contracts                  |
| `plugins/`        | 44 Python packages, one per provider. `plugins/stream` is the client for `acceleration/` |
| `sdks/swift/`     | Three iOS packages: `core` (state and API), `ui` (SwiftUI), `rtc` (voice over Stream Video) |
| `examples/agents/`| Runnable agents. `simple_voice_ai` is the smallest one                   |
| `sdks/go/`, `benchmark/` | The Go SDK and the voice benchmark. `go.work` ties the Go modules together |

## Local dev

Everything reads the repo-root `.env` for provider credentials.

The backend and dashboard together:

```bash
docker compose up --build
```

That serves the router on `:8080` and the dashboard on `:3000`, with Postgres on
`:55432` and Redis on `:56379`. Those two ports are also what the standalone `va-pg`
and `va-redis` containers use, so stop those first if they are running.

The router build fetches the private `getstream-go-webrtc` module over the host's SSH
agent, so `ssh-add -l` must show a key with access to it.

Without Docker, against the same Postgres and Redis:

```bash
go run ./cmd/router            # in acceleration/
npm run dev                    # in dashboard/, needs node >= 20.9
```

An agent, once the router is up:

```bash
cd examples/agents/simple_voice_ai
uv sync && uv run simple_voice_ai.py run
```

It prints a call URL on the dashboard; open it and talk.

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

`acceleration/api/openapi.yaml` is the source of truth for the HTTP layer. After editing it,
regenerate all three sides — see [acceleration/README.md](acceleration/README.md).

## Testing

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
