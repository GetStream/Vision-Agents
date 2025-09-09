## Overview
This project contains two parts, `examples` and `stream-agents/`.

Examples is our folder for testing different use-cases and models with the `Agent` while all
agent code including LLM support, turn detection, processors and observability lives in `stream-agents/`.

The backlog is managed on [Linear](https://linear.app/stream/project/agents-sdk-v1-1f1fd71f626f/issues) and groomed twice a week (Monday and Wednesday).
If a ticket is not assigned to anyone, please feel free to pick it up and share an update in #video_ai with the rest of the team.

## To install:
In the project root, run:
```bash
uv venv --python 3.12.2
uv sync --all-extras --dev
```

To setup your .env
```bash
cp env.example .env
```

## Running
```bash
uv run examples/01_simple_agent_example/01_simple_agent_example
```

## Tests

Integration test. (requires secrets in place)
```
uv run py.test -m "integration"
```

Everything other than integration

```
uv run py.test -m "not integration"
```

Plugin tests (TODO: not quite right. uv env is different for each plugin)

```
uv run py.test plugins/*/tests/*.py -m "not integration"
```

### Check

Shortcut to ruff, mypy and non integration tests:

```
uv run python dev.py check
```

### Formatting

```
uv run ruff check --fix
```

### Mypy type checks


```
uv run mypy --install-types --non-interactive -p stream_agents
```

```
uv run mypy --install-types --non-interactive --explicit-package-bases plugins
uv run mypy --install-types --non-interactive --explicit-package-bases plugins/xai
```

## Release

```
git tag -a v0.0.1 -m "Release 0.0.1" && git push --tags
```

## General Guidelines
1. We are experimenting and moving fast. Things may break, that is fine for now, but before merging to main, check that your code is running and if required, has tests.
2. Communication: Things are moving quickly, communicate what you're working on and what's blocking early and frequently in #video_ai
3. Avoid creating large PRs that's hard to review, break them up in to smaller reviewable PRs.
