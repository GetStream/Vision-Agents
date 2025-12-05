# Twilio Plugin

Twilio plugin for Vision Agents enabling voice call integration with real-time audio streaming.

## Features

- **Media Streaming**: Bidirectional audio streaming via Twilio Media Streams
- **Call Registry**: Track active calls with metadata (caller info, timestamps)
- **Audio Conversion**: Automatic mulaw/PCM conversion for Twilio compatibility
- **WebSocket Management**: Handle Twilio WebSocket connections

## Installation

```bash
uv add vision-agents[twilio]
```

## Usage

```python
from vision_agents.plugins import twilio

# Create a call registry to track active calls
registry = twilio.CallRegistry()

# When receiving a voice webhook, register the call
call = registry.create(call_sid="CA123...", form_data={"From": "+1234567890", ...})

# Create a media stream for the WebSocket connection
stream = twilio.MediaStream(websocket)
await stream.accept()

# Associate stream with call
call.twilio_stream = stream

# Run the stream (blocks until call ends)
await stream.run()
```

## Components

### TwilioCall

Dataclass representing an active call session:

```python
@dataclass
class TwilioCall:
    call_sid: str
    form_data: dict[str, Any]  # All Twilio webhook data
    twilio_stream: Optional[TwilioMediaStream]
    stream_call: Optional[Any]  # Stream video call
    started_at: datetime
    ended_at: Optional[datetime]
    
    # Convenience properties
    from_number: str  # Caller's phone number
    to_number: str    # Called phone number
    call_status: str  # Current call status
```

### TwilioCallRegistry

In-memory registry for managing active calls:

```python
registry = twilio.CallRegistry()
registry.create(call_sid, form_data)  # Register new call
registry.get(call_sid)                 # Look up call
registry.remove(call_sid)              # Remove and mark ended
registry.list_active()                 # List active calls
```

### TwilioMediaStream

Manages Twilio Media Stream WebSocket connections:

```python
stream = twilio.MediaStream(websocket)
await stream.accept()

# Access the audio track for publishing
stream.audio_track  # AudioStreamTrack at 8kHz

# Send audio back to Twilio
await stream.send_audio(pcm_data)

# Run until stream ends
await stream.run()
```

## Audio Utilities

```python
from vision_agents.plugins.twilio import mulaw_to_pcm, pcm_to_mulaw, TWILIO_SAMPLE_RATE

# Convert Twilio mulaw to PCM
pcm = mulaw_to_pcm(mulaw_bytes)

# Convert PCM to Twilio mulaw
mulaw = pcm_to_mulaw(pcm_data)
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TWILIO_SAMPLE_RATE` | Twilio audio sample rate | `8000` (8kHz) |

## Environment Variables

- `TWILIO_ACCOUNT_SID`: Your Twilio account SID
- `TWILIO_AUTH_TOKEN`: Your Twilio auth token

## Dependencies

- vision-agents
- twilio
- numpy


