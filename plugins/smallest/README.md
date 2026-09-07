# Smallest AI Plugin

This plugin provides STT and TTS capabilities using [Smallest AI](https://smallest.ai),
a low-latency speech platform with strong support for Indian languages.

## Features

- **STT**: WebSocket streaming speech-to-text (Pulse), 38 languages, 64ms
  time-to-first-transcript.
- **TTS**: WebSocket streaming text-to-speech (Lightning v3.1 / v3.1 Pro),
  ~100ms first audio chunk, with voice cloning support on the base model.

## Installation

```bash
uv add vision-agents-plugins-smallest
```

## Usage

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, openai, smallest, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Smallest AI"),
    instructions="You are a helpful voice assistant",
    llm=openai.LLM(model="gpt-4o-mini"),
    stt=smallest.STT(language="en"),
    tts=smallest.TTS(voice_id="magnus"),
    turn_detection=smart_turn.TurnDetection(),
)
```

Both services read the `SMALLEST_API_KEY` environment variable and send it
via the `Authorization: Bearer` header.

Pulse does not emit turn-boundary events on the streaming endpoint, so an
external `turn_detection` component (e.g. `smart_turn`) is required for
turn-taking.

## References

- [Smallest AI docs](https://docs.smallest.ai/)
- [Pulse realtime STT](https://docs.smallest.ai/waves/documentation/speech-to-text-pulse/realtime-web-socket/quickstart)
- [Lightning streaming TTS](https://docs.smallest.ai/waves/documentation/text-to-speech-lightning/http-vs-streaming-vs-web-sockets)
