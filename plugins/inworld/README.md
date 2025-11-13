# Inworld AI Text-to-Speech Plugin

A high-quality Text-to-Speech (TTS) plugin for Vision Agents that uses the Inworld AI API with streaming support.

## Installation

```bash
uv add vision-agents[inworld]
```

## Usage

```python
from vision_agents.plugins import inworld

# Initialize with API key from environment variable
tts = inworld.TTS()

# Or specify API key and other options directly
tts = inworld.TTS(
    api_key="your_inworld_api_key",
    voice_id="Dennis",
    model_id="inworld-tts-1",
    temperature=1.1
)

# Use with an Agent
from vision_agents.core import Agent
from vision_agents.plugins import getstream, gemini, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    tts=inworld.TTS(),
    llm=gemini.LLM("gemini-2.0-flash"),
    turn_detection=smart_turn.TurnDetection(),
)
```

## Configuration Options

- `api_key`: Inworld AI API key (default: reads from `INWORLD_API_KEY` environment variable)
- `voice_id`: The voice ID to use for synthesis (default: "Dennis")
- `model_id`: The model ID to use for synthesis. Options: "inworld-tts-1", "inworld-tts-1-max" (default: "inworld-tts-1")
- `temperature`: Determines the degree of randomness when sampling audio tokens. Accepts values between 0 and 2 (default: 1.1)

## Requirements

- Python 3.10+
- httpx>=0.27.0
  "av>=10.0.0",

## Getting Started

1. Get your Inworld AI API key from the [Inworld Portal](https://studio.inworld.ai/)
2. Set the `INWORLD_API_KEY` environment variable:
   ```bash
   export INWORLD_API_KEY="your_api_key_here"
   ```
3. Use the plugin in your Vision Agents application

## API Reference

The plugin implements the standard Vision Agents TTS interface:

- `stream_audio(text: str)`: Convert text to speech and return an async iterator of `PcmData` chunks
- `stop_audio()`: Stop audio playback (no-op for this plugin)
- `send(text: str)`: Send text to be converted to speech (inherited from base class)
