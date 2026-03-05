# Anthropic Plugin for Vision Agents

Anthropic Claude LLM integration for Vision Agents framework with support for streaming, function calling, and conversation memory.

It enables features such as:

- Streaming responses with Claude models
- Function calling capabilities for dynamic interactions
- Automatic conversation history management

## Installation

```bash
uv add vision-agents[anthropic]
# or directly
uv add vision-agents-plugins-anthropic
```

## Usage

### Standard LLM

This example shows how to use Claude with TTS and STT services for audio communication via `anthropic.LLM()` API.

The `anthropic.LLM()` class uses Anthropic's [Messages API](https://docs.anthropic.com/en/api/messages) under the hood.

```python
from vision_agents.core import User, Agent
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, cartesia, smart_turn, anthropic

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Friendly AI"),
    instructions="Be nice to the user",
    llm=anthropic.LLM("claude-sonnet-4-6"),
    tts=cartesia.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(),
)
```

## Function Calling

The `LLM` API supports function calling, allowing the assistant to invoke custom functions you define.

This enables dynamic interactions like:

- Database queries
- API calls to external services
- File operations
- Custom business logic

```python
from vision_agents.plugins import anthropic

llm = anthropic.LLM("claude-sonnet-4-6")


@llm.register_function(
    name="get_weather",
    description="Get the current weather for a given city"
)
def get_weather(city: str) -> dict:
    """Get weather information for a city."""
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny"
    }
# The function will be automatically called when the model decides to use it
```

## Requirements

- Python 3.10+
- GetStream account for video calls
- Anthropic API key

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)

## License

MIT
