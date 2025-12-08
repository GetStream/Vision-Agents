# OpenRouter Plugin

This plugin provides LLM capabilities using OpenRouter's API, which offers access to multiple LLM providers through a unified OpenAI-compatible interface. It enables developers to easily switch between different models from various providers (Anthropic, Google, OpenAI, etc.) without changing their code.

## Features

- Access to multiple LLM providers through a single API
- OpenAI-compatible interface for easy integration
- Support for various models including Claude, Gemini, GPT, and more
- Automatic conversion of instructions to system messages
- Manual conversation history management

## Installation

```bash
uv add vision-agents[openrouter]
```

## Usage

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import openrouter, getstream, elevenlabs, deepgram, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="OpenRouter AI"),
    instructions="Be helpful and friendly to the user",
    llm=openrouter.LLM(
        model="anthropic/claude-haiku-4.5",
    ),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(),
)
```

## Configuration

| Parameter | Description | Accepted Values |
|-----------|-------------|----------------|
| `api_key` | OpenRouter API key | `str \| None`. If not provided, uses `OPENROUTER_API_KEY` environment variable |
| `base_url` | OpenRouter API base URL | `str`. Default: `"https://openrouter.ai/api/v1"` |
| `model` | Model identifier to use | `str`. Default: `"openrouter/andromeda-alpha"`. Examples: `"anthropic/claude-haiku-4.5"`, `"google/gemini-2.5-flash"`, `"openai/gpt-4o"` |
| `**kwargs` | Additional arguments passed to OpenAI LLM | Any additional parameters supported by the underlying OpenAI LLM implementation |

## Dependencies

- vision-agents
- vision-agents-plugins-openai
