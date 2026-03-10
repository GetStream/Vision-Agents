# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Open Vision Agents — a Python monorepo (uv workspaces) for building real-time vision/voice AI agents. Core framework in `agents-core/`, 30+ plugins in `plugins/`.

### Running dev checks

Standard commands are in `CLAUDE.md` and `DEVELOPMENT.md`. Key quick-reference:

- **Full check:** `uv run --no-sync dev.py check` (ruff + mypy + validate-extras + unit tests)
- **Unit tests only:** `uv run --no-sync pytest -m "not integration"`
- **Lint:** `uv run --no-sync ruff check .` and `uv run --no-sync ruff format --check .`
- **Type check:** `uv run --no-sync mypy -p vision_agents`

### Gotchas

- Some tests (`test_agent_launcher.py`, `tests/test_agents/test_session_registry/`) require Docker for Redis testcontainers. These will error if Docker is not available. To skip them: `--ignore=tests/test_agents/test_agent_launcher.py --ignore=tests/test_agents/test_session_registry/`.
- Running actual agents requires external API keys (`STREAM_API_KEY`, `STREAM_API_SECRET`, plus an LLM provider key). These are only needed for integration tests and running examples, not for unit tests.
- Use `--no-sync` with `uv run` in sandboxed environments to avoid uv lockfile panics.
- The `fal` plugin workspace member does not appear in the `plugins/` directory listing but is referenced in `pyproject.toml` sources — this is harmless.
