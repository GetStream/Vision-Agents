# Mistral Voxtral STT Plugin

Mistral Voxtral realtime speech-to-text integration for Vision Agents.

## Features

- Real-time speech recognition via WebSocket streaming
- Low-latency transcription using Voxtral models
- Automatic language detection
- Partial transcript streaming for responsive UX

## Installation

```bash
uv add vision-agents-plugins-mistral
```

## Usage

```python
from vision_agents.plugins import mistral

stt = mistral.STT(
    api_key="your-api-key",  # Or set MISTRAL_API_KEY env var
    model="voxtral-mini-transcribe-realtime-2602",
)

await stt.start()

# Process audio chunks (called every ~20ms)
await stt.process_audio(pcm_data, participant)

# Subscribe to transcript events
@stt.events.subscribe
async def on_transcript(event: STTTranscriptEvent):
    print(f"Final: {event.text}")

@stt.events.subscribe
async def on_partial(event: STTPartialTranscriptEvent):
    print(f"Partial: {event.text}")

await stt.close()
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | Mistral API key | `MISTRAL_API_KEY` env var |
| `model` | Model identifier | `voxtral-mini-transcribe-realtime-2602` |
| `sample_rate` | Audio sample rate (Hz) | `16000` |
| `client` | Pre-configured Mistral client | `None` |

## Dependencies

- `mistralai[realtime]>=1.12.0`
- `vision-agents`
