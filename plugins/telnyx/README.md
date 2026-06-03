# Telnyx Plugin

Telnyx plugin for Vision Agents enabling voice call integration with real-time
media streaming.

## Features

- **Media Streaming**: Bidirectional audio streaming via Telnyx Media Streaming
- **Call Registry**: Track active calls with metadata and stream objects
- **Audio Conversion**: PCMU, PCMA, and L16 RTP payload conversion
- **WebSocket Management**: Handle Telnyx WebSocket media events

## Installation

```bash
uv add "vision-agents[telnyx]"
# or directly
uv add vision-agents-plugins-telnyx
```

## Usage

```python
from vision_agents.plugins import telnyx

# Create a call registry to track active calls
registry = telnyx.CallRegistry()

# Register a call from your Telnyx webhook handler
call = registry.create(call_control_id="v2:abc123", webhook_data={...})

# Create a media stream for the WebSocket connection
stream = telnyx.MediaStream(websocket)
await stream.accept()

# Associate stream with call
call.telnyx_stream = stream

# Run the stream until Telnyx sends a stop event
await stream.run()
```

## Components

### TelnyxCall

Dataclass representing an active call session:

```python
@dataclass
class TelnyxCall:
    call_control_id: str
    webhook_data: dict[str, Any] | None
    telnyx_stream: TelnyxMediaStream | None
    stream_call: Any | None
    started_at: datetime
    ended_at: datetime | None
```

### TelnyxCallRegistry

In-memory registry for managing active calls:

```python
registry = telnyx.CallRegistry()
registry.create(call_control_id, webhook_data=webhook_data)
registry.get(call_control_id)
registry.remove(call_control_id)
registry.list_active()
```

### TelnyxMediaStream

Manages Telnyx Media Streaming WebSocket connections:

```python
stream = telnyx.MediaStream(websocket)
await stream.accept()

# Access the audio track for publishing
stream.audio_track

# Send audio back to Telnyx when bidirectional RTP streaming is enabled
await stream.send_audio(pcm_data)

# Run until the stream ends
await stream.run()
```

To send audio back to the call, start Telnyx streaming with
`stream_bidirectional_mode=rtp`. The plugin supports PCMU and PCMA at 8 kHz, and
L16 at 16 kHz.

## Audio Utilities

```python
from vision_agents.plugins.telnyx import (
    pcma_to_pcm,
    pcm_to_pcma,
    pcm_to_pcmu,
    pcm_to_l16,
    pcmu_to_pcm,
    l16_to_pcm,
)

pcm = pcmu_to_pcm(payload)
payload = pcm_to_pcmu(pcm)
```

## Configuration

| Parameter                    | Description                         | Default |
|------------------------------|-------------------------------------|---------|
| `TELNYX_DEFAULT_SAMPLE_RATE` | Telnyx PCMU and PCMA sample rate    | `8000`  |
| `TELNYX_L16_SAMPLE_RATE`     | Telnyx L16 bidirectional sample rate | `16000` |

## Environment Variables

- `TELNYX_API_KEY`: Your Telnyx API key for Call Control API requests

## Dependencies

- vision-agents
- numpy
- fastapi
