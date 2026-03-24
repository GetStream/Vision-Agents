# Monty Agent Example

A voice AI agent powered by Gemini Realtime that can write and execute Python code in a secure sandbox ([Monty](https://github.com/pydantic/monty)).

The agent can do calculations, fetch data from APIs, and search the web — all through voice conversation.

## Quick start

```bash
uv run python monty_agent_example.py run
```

## Try asking

- "What is 2 to the power of 100?"
- "What's the current Bitcoin price?"
- "Search for the latest James Webb telescope discoveries"
- "What's the weather in London?"

## How it works

The agent has one tool — `run_python_code`. When a question needs computation or data, Gemini writes Python code and executes it in the Monty sandbox. The sandbox has access to:

| Function | Description |
|----------|-------------|
| `await fetch(url)` | HTTP requests |
| `json_parse(text)` | Parse JSON (no `import json` in sandbox) |
| `json_dumps(obj)` | Serialize to JSON |
| `await web_search(query)` | Web search |
| `import math` | Math module |
| `import re` | Regex module |

## Web search

By default the agent uses **DuckDuckGo Lite** for web search — no setup needed, works out of the box. It parses HTML from DuckDuckGo's lite interface.

For better results, you can switch to **[SearXNG](https://github.com/searxng/searxng)** — an open-source metasearch engine that aggregates results from Google, Brave, DuckDuckGo and others, and returns clean JSON.

### Setting up SearXNG (optional)

```bash
docker run -d -p 8080:8080 --name searxng searxng/searxng
```

Enable JSON API (disabled by default):

```bash
docker exec searxng sed -i '/^  formats:/,/^[^ ]/{s/    - html/    - html\n    - json/}' /etc/searxng/settings.yml
docker restart searxng
```

Then change one line in `monty_agent_example.py`:

```python
# Before (default)
set_search_engine(DuckDuckGoLite())

# After
set_search_engine(SearXNG())
# or with custom URL:
set_search_engine(SearXNG(base_url="http://my-searxng:8080"))
```

### DuckDuckGo Lite vs SearXNG

| | DuckDuckGo Lite | SearXNG |
|---|---|---|
| Setup | None | Docker one-liner |
| Results quality | Basic (HTML parsing, single source) | Rich (multiple engines, scored, ranked) |
| Reliability | Fragile (HTML structure may change) | Stable (JSON API) |
| Speed | Fast | Fast (local) |
| Privacy | DuckDuckGo servers | Fully local |

## Project structure

```
monty_agent_example.py  — entry point, agent setup + sandbox runner
instructions.py         — agent instructions and tool description
sandbox/
  __init__.py           — EXTERNAL_FUNCTIONS, TYPE_STUBS
  tools.py              — fetch, json_parse, json_dumps, web_search
  search.py             — SearchEngine protocol + DuckDuckGoLite + SearXNG
  stubs.py              — type definitions for Monty's type checker
```
