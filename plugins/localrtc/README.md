# Vision Agents - Local RTC Plugin

Local RTC plugin for Vision Agents, enabling local audio/video stream processing without external RTC infrastructure.

## Installation

```bash
uv add vision-agents-plugins-localrtc
```

## Features

- Local audio capture and playback via sounddevice
- Video capture via OpenCV
- Optional GStreamer pipelines for embedded systems (Raspberry Pi, custom hardware)
- Automatic audio format negotiation with LLM providers
- Compatible with Vision Agents Edge Transport interface

## Dependencies

- sounddevice: Local audio capture and playback
- numpy: Audio data processing
- aiortc: WebRTC compatibility layer
- opencv-python: Video capture

Optional:
- PyGObject + GStreamer: Custom pipelines for embedded systems

## Quick Start

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import localrtc, gemini

# Create local edge transport
edge = localrtc.Edge(
    audio_device="default",
    speaker_device="default",
    video_device=0,
    sample_rate=16000,
    channels=1,
)

# Create agent
agent = Agent(
    edge=edge,
    agent_user=User(name="Local AI", id="agent"),
    instructions="You're a helpful voice AI assistant.",
    llm=gemini.Realtime(),
)

# Run
call = await agent.create_call("default", "my-call")
async with agent.join(call):
    await agent.simple_response("Say hello!")
    await agent.finish()
```

## Configuration

Configuration is provided programmatically via dataclasses.

### Basic Usage

```python
from vision_agents.plugins.localrtc import Edge

# Use defaults
edge = Edge()

# Or specify devices
edge = Edge(
    audio_device="default",
    speaker_device="default",
    video_device=0,
    sample_rate=16000,
    channels=1,
)
```

### Custom Configuration

```python
from vision_agents.plugins.localrtc.localedge import (
    LocalEdge,
    LocalEdgeConfig,
    AudioConfig,
    VideoConfig,
)

config = LocalEdgeConfig(
    audio=AudioConfig(
        input_sample_rate=48000,
        output_sample_rate=24000,
        input_channels=2,
        output_channels=1,
        bit_depth=16,
    ),
    video=VideoConfig(
        default_width=1280,
        default_height=720,
        default_fps=30,
    ),
)

edge = LocalEdge(config=config)
```

### Configuration Reference

#### AudioConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_sample_rate` | int | 16000 | Audio input sampling rate in Hz |
| `output_sample_rate` | int | 24000 | Output sample rate (auto-negotiated with LLM) |
| `input_channels` | int | 1 | Number of audio input channels |
| `output_channels` | int | 1 | Number of audio output channels |
| `bit_depth` | int | 16 | Audio bit depth (8, 16, 24, or 32) |
| `input_buffer_duration` | float | 2.0 | Input buffer duration in seconds |
| `output_buffer_size_ms` | int | 10000 | Output buffer size in milliseconds |
| `capture_chunk_duration` | float | 0.1 | Audio capture chunk duration in seconds |
| `playback_chunk_duration` | float | 0.05 | Audio playback chunk duration in seconds |

#### VideoConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_width` | int | 640 | Video frame width in pixels |
| `default_height` | int | 480 | Video frame height in pixels |
| `default_fps` | int | 30 | Frames per second |
| `format` | str | "BGR" | Video format (BGR for OpenCV compatibility) |
| `max_buffers` | int | 1 | Maximum buffers for GStreamer appsink |

## Device Discovery

```python
from vision_agents.plugins.localrtc import Edge

devices = Edge.list_devices()

# Audio inputs (microphones)
for device in devices["audio_inputs"]:
    print(f"{device['index']}: {device['name']}")

# Audio outputs (speakers)
for device in devices["audio_outputs"]:
    print(f"{device['index']}: {device['name']}")

# Video inputs (cameras) - not yet implemented
for device in devices["video_inputs"]:
    print(f"{device['index']}: {device['name']}")
```

Use devices by index or name:

```python
edge = Edge(
    audio_device=1,                    # By index
    speaker_device="Built-in Speakers", # By name
    video_device=0,
)
```

## Audio Format Negotiation

Output audio format is automatically negotiated with the LLM provider during `agent.join()`:

- **Gemini Realtime**: 24kHz mono
- **OpenAI Realtime**: 24kHz mono
- **Default**: 24kHz mono

Input format is configured via `sample_rate` and `channels` parameters.

## GStreamer Pipelines (Optional)

For embedded systems like Raspberry Pi, use custom GStreamer pipelines:

```python
edge = Edge(
    custom_pipeline={
        "audio_source": "alsasrc device=hw:1,0 ! audioconvert ! audioresample",
        "video_source": "v4l2src device=/dev/video0 ! videoconvert",
        "audio_sink": "alsasink device=hw:0,0",
    },
    sample_rate=16000,
    channels=1,
)
```

Requires GStreamer installation:

```bash
# Ubuntu/Debian
sudo apt-get install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-base

# macOS
brew install pygobject3 gstreamer
```

## Troubleshooting

### No Audio Input

1. Check device availability:
   ```python
   devices = Edge.list_devices()
   print(devices["audio_inputs"])
   ```

2. Verify device permissions (Linux):
   ```bash
   groups  # Should include 'audio'
   sudo usermod -a -G audio $USER
   ```

### No Audio Output

1. Verify speaker device:
   ```python
   devices = Edge.list_devices()
   print(devices["audio_outputs"])
   ```

2. Check system volume and mute settings

### GStreamer Errors

If using custom pipelines:

```bash
# Test audio
gst-launch-1.0 audiotestsrc ! autoaudiosink

# Test video
gst-launch-1.0 videotestsrc ! autovideosink

# List ALSA devices
arecord -L
aplay -L
```
