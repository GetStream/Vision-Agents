# CLAUDE.md

## Project overview

Python monorepo managed with **uv workspaces**.
The core framework lives in `agents-core/` and plugins live in `plugins/` (37+ packages).
Python >= 3.10, 3.12 recommended.

## Commands

All commands use `uv`. Never use `python -m`. If you run into dependency issues, stop and ask.

```bash
# Full check (ruff + mypy + unit tests)
uv run dev.py check

# Unit tests only
uv run py.test -m "not integration" -n auto

# Integration tests (needs .env secrets)
uv run py.test -m "integration"

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy
```

## Testing

- Framework: pytest. Never mock.
- `@pytest.mark.asyncio` is not needed (asyncio_mode = auto).
- Integration tests use `@pytest.mark.integration`.
- Never adjust `sys.path`.

## Python rules

- Never use `from __future__ import annotations`.
- Never write `except Exception as e`. Catch specific exceptions.
- Avoid `getattr`, `hasattr`, `delattr`, `setattr`; prefer normal attribute access.
- Docstrings: Google style, keep them short.
- Do not use section comments like `# -- some section --`
- Prefer `logger.exception()` when logging an error with a traceback instead of `logger.error("Error: {exc}")`

## Code style

**Imports**:

- ordered as: stdlib, third-party, local package, relative. Use `TYPE_CHECKING` guard for imports only needed by type annotations.

**Naming**:

- private attributes and methods use a leading underscore (`_sessions`, `_warmup_agent`). Public API is plain snake_case.

**Type annotations**:

- use them everywhere. Modern syntax: `X | Y` unions, `dict[str, T]` generics, full `Callable` signatures, `Optional` for nullable params.

**Logging**:
module-level `logger = logging.getLogger(__name__)`. Use `debug` for lifecycle, `info` for notable events, `error` for failures without a traceback,
`exception` for errors with traceback.

**Constructor validation**:

- raise `ValueError` with a descriptive message for invalid args. Prefer custom domain exceptions over generic ones.

**Async patterns**:

- async-first lifecycle methods (`start`/`stop`). Support `__aenter__`/`__aexit__` for context manager usage.
- Use `asyncio.Lock`, `asyncio.Task`, `asyncio.gather` for concurrency.
- Clean up resources in `finally` blocks.

**Method order**:

- `__init__`, public lifecycle methods, properties, public feature methods, private helpers, dunder methods.
