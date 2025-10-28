# OpenRouter Plugin

OpenRouter plugin for vision agents. This plugin provides LLM capabilities using OpenRouter's API, which is compatible with the OpenAI API format.

## Note/ Issues

Instruction following doesn't always work with openrouter atm.

## Installation

```bash
uv pip install vision-agents-plugins-openrouter
```

## Usage

```python
from vision_agents.plugins import openrouter, getstream, elevenlabs, cartesia, deepgram, smart_turn


agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="OpenRouter AI"),
    instructions="Be helpful and friendly to the user",
    llm=openrouter.LLM(
        model="anthropic/claude-haiku-4.5",  # Can also use other models like anthropic/claude-3-opus
    ),
    tts=elevenlabs.TTS(),
    stt=deepgram.STT(),
    turn_detection=smart_turn.TurnDetection(
        buffer_duration=2.0, confidence_threshold=0.5
    )
)
```
