# Local Transport Example

This example demonstrates how to run a vision agent using local audio/video I/O (microphone, speakers, and camera) instead of a cloud-based edge network.

## Overview

The LocalTransport provides:
- **Microphone input**: Captures audio from your microphone
- **Speaker output**: Plays AI responses on your speakers
- **Camera input**: Captures video from your camera (optional)
- **No cloud dependencies**: Media runs locally (except for the LLM, TTS, and STT services)

## Examples

There are two example scripts:

### 1. Basic Voice Agent (`local_transport_example.py`)

Uses Gemini LLM with Deepgram STT and ElevenLabs TTS for a voice-only experience.

```bash
uv run python local_transport_example.py
```

### 2. Vision Agent with Gemini Realtime (`local_transport_realtime_example.py`)

Uses Gemini Realtime for native audio/video understanding. This lets Gemini see through your camera!

```bash
uv run python local_transport_realtime_example.py
```

Try asking: "What do you see?" or "Describe what's in front of you"

## Prerequisites

1. A working microphone and speakers
2. A camera (optional for basic example, recommended for realtime example)
3. API keys:

### For basic example:
- Google AI (for Gemini LLM)
- Deepgram (for STT)
- ElevenLabs (for TTS)

### For realtime example:
- Google AI (for Gemini Realtime) - handles audio/video natively

## Setup

1. Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

2. Install dependencies:

```bash
cd examples/09_local_transport_example
uv sync
```

## Device Selection

Both examples will prompt you to select:
1. **Input device** (microphone)
2. **Output device** (speakers)
3. **Video device** (camera) - can be skipped by entering 'n'

Press Enter to use the default device, or enter a number to select a specific device.

Press `Ctrl+C` to stop the agent.

## Listing Audio Devices

To see available audio devices on your system:

```python
from vision_agents.core.edge.local_transport import list_audio_devices
list_audio_devices()
```

## Configuration

You can customize the audio settings when creating the LocalTransport:

```python
from vision_agents.core.edge.local_transport import LocalTransport

transport = LocalTransport(
    sample_rate=48000,       # Audio sample rate (Hz)
    input_channels=1,        # Mono microphone input
    output_channels=2,       # Stereo speaker output
    blocksize=1024,          # Audio buffer size
    input_device=None,       # Default input device (or device index)
    output_device=None,      # Default output device (or device index)
)
```

## Troubleshooting

### No audio input/output

1. Check that your microphone and speakers are properly connected
2. Run `list_audio_devices()` to see available devices
3. Try specifying explicit device indices in the LocalTransport constructor

### Audio quality issues

- Try increasing the `blocksize` parameter for smoother audio
- Ensure your microphone isn't picking up too much background noise

### Permission errors

On macOS, you may need to grant microphone permissions to your terminal application.
