# Gradium Plugin

Gradium plugin for Vision Agents providing low-latency, high-quality speech-to-text (STT) and text-to-speech (TTS) capabilities using the official Gradium SDK.

## Features

- **Low-latency STT & TTS**: Time-to-first-token below 300ms with streaming support
- **Multilingual**: Supports English, French, German, Spanish, and Portuguese
- **Voice Activity Detection**: Built-in VAD for automatic turn detection in STT
- **Voice Library**: Multiple voices to choose from, plus instant voice cloning
- **Speed Control**: Adjustable speech speed for TTS
- **Official SDK**: Uses the official `gradium` Python package

## Installation

```bash
uv add vision-agents[gradium]
```

## Usage

### Speech-to-Text (STT)

```python
from vision_agents.plugins import gradium

# Create Gradium STT instance
stt = gradium.STT(
    language="en",      # en, fr, de, es, pt
    vad_threshold=0.7,  # VAD sensitivity (0.0-1.0)
)

# Start the stream
await stt.start()

# Process audio
await stt.process_audio(pcm_data)

# Close when done
await stt.close()
```

### Text-to-Speech (TTS)

```python
from vision_agents.plugins import gradium

# Create Gradium TTS instance
tts = gradium.TTS(
    voice_id="YTpq7expH9539ERJ",  # Voice from library or custom voice
    speed=0.0,                     # -4.0 (faster) to 4.0 (slower)
)

# Convert text to speech
await tts.send("Hello from Gradium!")
```

### Using a Shared GradiumClient

```python
import gradium
from vision_agents.plugins import gradium as gradium_plugin

# Create a shared client
client = gradium.client.GradiumClient(api_key="your-api-key")

# Use the same client for both STT and TTS
stt = gradium_plugin.STT(client=client)
tts = gradium_plugin.TTS(client=client)
```

### Using with Agent

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gradium

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Assistant", id="agent"),
    instructions="Be helpful",
    stt=gradium.STT(language="en"),
    tts=gradium.TTS(),
)
```

## Configuration

### STT Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | Gradium API key (or set `GRADIUM_API_KEY` env var) | Required |
| `language` | Language code: `en`, `fr`, `de`, `es`, `pt` | `"en"` |
| `vad_threshold` | VAD inactivity probability threshold (0.0-1.0) | `0.7` |
| `client` | Optional pre-configured `GradiumClient` instance | `None` |

### TTS Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_key` | Gradium API key (or set `GRADIUM_API_KEY` env var) | Required |
| `voice_id` | Voice ID from library or custom voice | `"YTpq7expH9539ERJ"` |
| `model_name` | TTS model to use | `"default"` |
| `speed` | Speed control: -4.0 (faster) to 4.0 (slower) | `0.0` |
| `client` | Optional pre-configured `GradiumClient` instance | `None` |

### TTS Output Format

When using PCM output, audio is delivered as:
- **Sample Rate**: 48kHz
- **Format**: 16-bit signed integer
- **Channels**: Mono
- **Chunk Size**: 3840 samples (80ms)

### Adding Pauses in TTS

Use the `<break>` tag to add pauses:

```python
text = "Hello. <break time=\"1.5s\" /> How are you today?"
await tts.send(text)
```

Break time must be between 0.1s and 2.0s, with spaces before and after the tag.

## Environment Variables

- `GRADIUM_API_KEY`: Your Gradium API key

## Dependencies

- vision-agents
- gradium (official Gradium Python SDK)

## API Reference

For more details, see the [Gradium API Documentation](https://gradium.ai/api_docs.html).
