# MiniMax Plugin

This plugin provides LLM capabilities using MiniMax's OpenAI-compatible API. MiniMax offers high-performance language models with 204,800 token context windows.

## Features

- OpenAI-compatible Chat Completions API
- Streaming responses with real-time chunk events
- Function/tool calling support
- Conversation history management

## Installation

```bash
uv add "vision-agents[minimax]"
# or directly
uv add vision-agents-plugins-minimax
```

## Usage

```python
from vision_agents.plugins import minimax

llm = minimax.LLM(model="MiniMax-M2.5")
response = await llm.create_response(
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.text)
```

## Configuration

| Parameter  | Description                | Accepted Values                                                                                          |
|------------|----------------------------|----------------------------------------------------------------------------------------------------------|
| `api_key`  | MiniMax API key            | `str \| None`. If not provided, uses `MINIMAX_API_KEY` environment variable                              |
| `base_url` | MiniMax API base URL       | `str \| None`. Default: `"https://api.minimax.io/v1"`. Use `"https://api.minimaxi.com/v1"` for China     |
| `model`    | Model identifier to use    | `str`. Default: `"MiniMax-M2.5"`. Also available: `"MiniMax-M2.5-highspeed"`                             |
| `client`   | Custom AsyncOpenAI client  | `AsyncOpenAI \| None`. For dependency injection or custom configuration                                  |

## Supported Models

| Model                    | Description                                      |
|--------------------------|--------------------------------------------------|
| `MiniMax-M2.5`           | Peak Performance. Ultimate Value. Master the Complex |
| `MiniMax-M2.5-highspeed` | Same performance, faster and more agile          |

Both models support a 204,800 token context window.

## API Documentation

- [OpenAI Compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api)

## Dependencies

- vision-agents
- openai
