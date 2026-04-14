# Sarvam AI Plugin

This plugin provides STT, TTS, and LLM capabilities using Sarvam AI, a suite of
AI models built for Indian languages.

## Features

- **STT**: WebSocket streaming speech-to-text (Saarika / Saaras) with Voice
  Activity Detection for turn events.
- **TTS**: WebSocket streaming text-to-speech (Bulbul) with configurable
  speaker, pace, and language.
- **LLM**: OpenAI-compatible chat completions (Sarvam-30B / Sarvam-105B /
  Sarvam-M) via the existing `ChatCompletionsLLM` from the OpenAI plugin.

## Installation

```bash
uv add vision-agents-plugins-sarvam
```

## Usage

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, sarvam, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Sarvam AI"),
    instructions="Reply in Hindi or English, whichever the user speaks",
    llm=sarvam.LLM(model="sarvam-30b"),
    stt=sarvam.STT(language="hi-IN"),
    tts=sarvam.TTS(speaker="shubh"),
    turn_detection=smart_turn.TurnDetection(),
)
```

All three services read the same `SARVAM_API_KEY` environment variable and send
it via the `api-subscription-key` header.

## References

- [Sarvam API docs](https://docs.sarvam.ai/)
