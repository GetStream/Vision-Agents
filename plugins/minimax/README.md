# MiniMax Plugin for Vision Agents

The MiniMax plugin adds the [MiniMax](https://platform.minimax.io) family of language models as a first-class LLM provider for [Vision Agents](https://visionagents.ai). It uses the OpenAI-compatible Chat Completions API at `https://api.minimax.io/v1`, so the implementation inherits the streaming and tool-calling behavior of the OpenAI Chat Completions plugin.

## Installation

```bash
uv add "vision-agents-plugins-minimax"
```

## Configuration

Set your MiniMax API key (sign up at <https://platform.minimax.io>):

```bash
export MINIMAX_API_KEY=...
```

The plugin also honors `MINIMAX_BASE_URL` for users that proxy the API.

## Models

| Model | Description |
|-------|-------------|
| `MiniMax-M3` (default) | Latest flagship model. 512K context window, up to 128K output, supports image input. |
| `MiniMax-M2.7` | Previous generation flagship model. |
| `MiniMax-M2.7-highspeed` | Low-latency variant of M2.7. |

## Usage

```python
from vision_agents.plugins import minimax

llm = minimax.LLM()  # defaults to MiniMax-M3
llm = minimax.LLM(model="MiniMax-M2.7")
```

## Notes

- Default base URL is `https://api.minimax.io/v1` (overseas). Do not use `api.minimax.chat` for production traffic.
- Default `temperature` is `1.0` (the API rejects `0.0`).
- The `response_format` field is not supported by the MiniMax Chat Completions API and is intentionally not exposed.
