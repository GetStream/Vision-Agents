# Deepgram Plugin

Speech-to-Text (STT) and Text-to-Speech (TTS) plugins for Vision Agents using the Deepgram API.

## Installation

```bash
uv add "vision-agents[deepgram]"
# or directly
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

Low-latency text-to-speech using Deepgram's Flux TTS via WebSocket streaming (`/v2/speak`).

```python
from vision_agents.plugins import deepgram

tts = deepgram.TTS(
    model="flux-haley-en",  # Default voice
    sample_rate=16000,  # Audio sample rate
    speed=1.0,  # Optional speech-rate multiplier (0.85–1.15)
)
```

### Available Voices

Deepgram Flux voices use the `flux-{voice}-en` format. Aura model strings are not supported.

- `flux-haley-en` - Default featured voice
- `flux-kit-en`
- See [Flux TTS voices](https://developers.deepgram.com/docs/flux-tts/voices) for all options

### TTS Docs

- https://developers.deepgram.com/docs/flux-tts/overview
- https://developers.deepgram.com/docs/flux-tts/voices

## Environment Variables

Set `DEEPGRAM_API_KEY` in your environment or pass `api_key` to the constructor.

## Example

See the [example](./example/) directory for a complete working example using both STT and TTS.
