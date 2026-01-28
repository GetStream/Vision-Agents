# Local Transport Example

This example demonstrates how to run a vision agent using local audio I/O (microphone and speakers) instead of a cloud-based edge network.

## Overview

The LocalTransport provides:
- **Microphone input**: Captures audio from your default microphone
- **Speaker output**: Plays AI responses on your default speakers
- **No cloud dependencies**: Everything runs locally (except for the LLM, TTS, and STT services)

## Prerequisites

1. A working microphone and speakers
2. API keys for the following services:
   - Google AI (for Gemini LLM)
   - Deepgram (for STT)
   - ElevenLabs (for TTS)

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

## Running the Example

```bash
uv run python local_transport_example.py
```

The agent will:
1. Start listening on your microphone
2. Transcribe your speech using Deepgram
3. Generate responses using Gemini
4. Speak responses using ElevenLabs TTS
5. Play the audio on your speakers

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
