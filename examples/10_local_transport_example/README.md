# Local Transport Example

This example demonstrates how to run a vision agent using local audio/video I/O (microphone, speakers, and camera) instead of a cloud-based edge network.

## Overview

The LocalEdge provides:

- **Microphone input**: Captures audio from your microphone
- **Speaker output**: Plays AI responses on your speakers
- **Camera input**: Captures video from your camera (optional)
- **No cloud dependencies**: Media runs locally (except for the LLM, TTS, and STT services)

## Running

Uses Gemini LLM with Deepgram STT and TTS for a voice experience with optional camera input.

```bash
uv run python local_transport_example.py
```

## Prerequisites

1. A working microphone and speakers
2. A camera (optional, for video input)
3. API keys:
   - Google AI (for Gemini LLM)
   - Deepgram (for STT and TTS)

## Setup

1. Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

2. Install dependencies:

```bash
cd examples/10_local_transport_example
uv sync
```

## Device Selection

The example will prompt you to select:

1. **Input device** (microphone)
2. **Output device** (speakers)
3. **Video device** (camera) - can be skipped by entering 'n'

Press Enter to use the default device, or enter a number to select a specific device.

Press `Ctrl+C` to stop the agent.

## Listing Audio Devices

To see available audio devices on your system:

```python
from vision_agents.plugins.local.devices import list_audio_input_devices, list_audio_output_devices

list_audio_input_devices()
list_audio_output_devices()
```

## Configuration

You can customize the audio settings when creating the LocalEdge:

```python
from vision_agents.plugins.local import LocalEdge
from vision_agents.plugins.local.devices import (
    select_audio_input_device,
    select_audio_output_device,
)

input_device = select_audio_input_device()
output_device = select_audio_output_device()

edge = LocalEdge(
    audio_input=input_device,  # AudioInputDevice (microphone)
    audio_output=output_device,  # AudioOutputDevice (speakers)
)
```

## Troubleshooting

### No audio input/output

1. Check that your microphone and speakers are properly connected
2. Run `list_audio_input_devices()` or `list_audio_output_devices()` to see available devices
3. Try specifying explicit device indices in the LocalEdge constructor

### Audio quality issues

- Try increasing the `blocksize` parameter for smoother audio
- Ensure your microphone isn't picking up too much background noise

### Permission errors

On macOS, you may need to grant microphone permissions to your terminal application.
