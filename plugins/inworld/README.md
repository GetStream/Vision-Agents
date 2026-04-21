# Inworld AI Plugin

Inworld AI integration for Vision Agents. Provides both text-to-speech and
a WebRTC-based Realtime speech-to-speech conversational API.

## Installation

```bash
uv add "vision-agents[inworld]"
# or directly
uv add vision-agents-plugins-inworld
```

Get your API key from the [Inworld Portal](https://studio.inworld.ai/) and set
`INWORLD_API_KEY` in your environment (or pass `api_key=` explicitly).

## TTS

High-quality text-to-speech with streaming support.

```python
from vision_agents.plugins import inworld

tts = inworld.TTS()

# Or specify explicitly
tts = inworld.TTS(
    api_key="your_inworld_api_key",
    voice_id="Dennis",
    model_id="inworld-tts-1.5-max",
    temperature=1.1,
)
```

### TTS options

- `api_key`: Inworld AI API key (default: reads from `INWORLD_API_KEY`)
- `voice_id`: Voice to use (default: `"Dennis"`)
- `model_id`: `"inworld-tts-1.5-max"`, `"inworld-tts-1.5-mini"`, `"inworld-tts-1"`, `"inworld-tts-1-max"` (default: `"inworld-tts-1.5-max"`)
- `temperature`: 0–2 (default: 1.1)

## Realtime (WebRTC)

Low-latency speech-to-speech via Inworld's Realtime API. This transport uses
WebRTC (UDP, native Opus) for lower latency than the WebSocket alternative.
Requires a WebRTC-capable edge transport — pair with `getstream.Edge()` as
shown below.

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, inworld, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="My Agent", id="agent"),
    llm=inworld.Realtime(
        model="openai/gpt-4o-mini",
        voice="Dennis",
        instructions="You are a friendly voice assistant.",
    ),
    turn_detection=smart_turn.TurnDetection(),
)
```

### Realtime options

- `model`: provider-prefixed model ID. Examples: `"openai/gpt-4o-mini"` (default), `"google-ai-studio/gemini-2.5-flash"`, `"inworld/<router-id>"` for an Inworld router
- `voice`: voice for audio responses (default: `"Dennis"`; `"Clive"`, `"Olivia"` and custom voices also supported)
- `api_key`: Inworld AI API key (default: reads from `INWORLD_API_KEY`)
- `instructions`: system prompt
- `realtime_session`: advanced — pass a full `RealtimeSessionCreateRequestParam` for session fields not exposed by the primary args (custom turn-detection, `tool_choice`, etc.)

### Registering tools

```python
realtime = inworld.Realtime()

@realtime.register_function(description="Get the current weather for a city.")
async def get_weather(city: str) -> str:
    return f"It's sunny in {city}."
```

Tools follow the OpenAI function-calling schema. Inworld's Realtime API is
protocol-compatible with OpenAI's Realtime API, so registered functions flow
through the same `response.function_call_arguments.done` path.

### Notes

- v1 is WebRTC only; a WebSocket transport may be added later.
- Video input is not currently supported by Inworld's Realtime API.

## Requirements

- Python 3.10+
- `httpx>=0.28`, `av>=10`, `aiortc>=1.9`, `openai[realtime]>=2.26,<3`
