# Requesty Plugin

This plugin provides LLM capabilities using [Requesty](https://requesty.ai/), an OpenAI-compatible LLM gateway/router that offers access to multiple LLM providers through a single API. It enables developers to easily switch between different models from various providers (OpenAI, Anthropic, Google, DeepSeek, etc.) without changing their code.

## Features

- Access to multiple LLM providers through a single API
- OpenAI-compatible interface for easy integration
- Support for various models including GPT, Claude, Gemini, DeepSeek, and more
- Automatic conversion of instructions to system messages
- Manual conversation history management

## Installation

```bash
uv add "vision-agents[requesty]"
# or directly
uv add vision-agents-plugins-requesty
```

## Usage

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import requesty, getstream, elevenlabs, deepgram, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Requesty AI"),
    instructions="Be helpful and friendly to the user",
    llm=requesty.LLM(
        model="anthropic/claude-sonnet-4-5",
    ),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(),
)
```

Set your API key via the `REQUESTY_API_KEY` environment variable (get one at
[app.requesty.ai/api-keys](https://app.requesty.ai/api-keys)). Models are named
`provider/model`; browse the catalog at
[app.requesty.ai/router/list](https://app.requesty.ai/router/list).

## Configuration

| Parameter  | Description                               | Accepted Values                                                                                                                          |
|------------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `api_key`  | Requesty API key                          | `str \| None`. If not provided, uses `REQUESTY_API_KEY` environment variable                                                             |
| `base_url` | Requesty API base URL                     | `str`. Default: `"https://router.requesty.ai/v1"`                                                                                       |
| `model`    | Model identifier to use                   | `str`. Default: `"openai/gpt-4o-mini"`. Examples: `"anthropic/claude-sonnet-4-5"`, `"google/gemini-2.5-flash"`, `"openai/gpt-4o"`        |
| `max_tokens` | Upper limit on generated tokens         | `int \| None`                                                                                                                            |
| `tools_max_rounds` | Max rounds for multi-hop tool calls | `int`. Default: `3`                                                                                                                      |

## Dependencies

- vision-agents
- openai
