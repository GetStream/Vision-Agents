# Deepgram Plugin

Speech-to-Text (STT) and Text-to-Speech (TTS) plugins for Vision Agents using the Deepgram API.

## Installation

```bash
uv add vision-agents-plugins-deepgram
```

## Speech-to-Text (STT)

High-quality speech recognition using Deepgram's Flux model with built-in turn detection.

```python
from vision_agents.plugins import deepgram

stt = deepgram.STT(
    model="flux-general-en",  # Default model
    eager_turn_detection=True,  # Enable eager end-of-turn detection
)
```

### STT Docs

- https://developers.deepgram.com/docs/flux/quickstart
- https://github.com/deepgram/deepgram-python-sdk/blob/main/examples/listen/v2/connect/async.py

## Text-to-Speech (TTS)

Low-latency text-to-speech using Deepgram's Aura model via WebSocket streaming.

```python
from vision_agents.plugins import deepgram

tts = deepgram.TTS(
    model="aura-2-thalia-en",  # Default voice
    sample_rate=16000,  # Audio sample rate
)
```

### Available Voices

Deepgram offers various Aura voice models:
- `aura-2-thalia-en` - Default female voice
- `aura-2-orion-en` - Male voice
- See [TTS Models](https://developers.deepgram.com/docs/tts-models) for all options

### TTS Docs

- https://developers.deepgram.com/docs/tts-websocket
- https://developers.deepgram.com/docs/streaming-text-to-speech

## Environment Variables

Set `DEEPGRAM_API_KEY` in your environment or pass `api_key` to the constructor.

## Example

See the [example](./example/) directory for a complete working example using both STT and TTS.
