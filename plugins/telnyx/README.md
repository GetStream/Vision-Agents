# Telnyx Plugin

Telnyx plugin for Vision Agents enabling inbound and outbound phone calls with
real-time bidirectional media streaming.

## Features

- **Media Streaming**: Bidirectional audio streaming via Telnyx Media Streaming
- **Call Control**: Support for programmable inbound and outbound phone calls
- **Call Registry**: Track active calls with metadata, stream objects, and
  validation tokens
- **Audio Conversion**: PCMU, PCMA, and L16 RTP payload conversion
- **WebSocket Management**: Handle Telnyx WebSocket media events
- **Stream Bridge**: Attach a Telnyx phone participant to a Stream call

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
call = registry.create(
    call_control_id="v2:abc123",
    webhook_data={"data": {"payload": {"from": "+15551234567"}}},
)

# Create a media stream for the WebSocket connection
stream = telnyx.MediaStream(websocket)
await stream.accept()

# Associate stream with call
call.telnyx_stream = stream

# Run the stream until Telnyx sends a stop event
await stream.run()
```

## Examples

See [examples/](examples/) for minimal inbound and outbound Telnyx phone
examples.

```bash
# Outbound call
uv run plugins/telnyx/examples/outbound_call.py \
  --setup-telnyx \
  --from +15551234567 \
  --to +15557654321

# Inbound call server
uv run plugins/telnyx/examples/inbound_call.py \
  --setup-telnyx \
  --phone-number +15551234567
```

Telnyx phone calls require a Call Control App. The Call Control App is where
Telnyx sends call webhooks such as `call.initiated`, `call.answered`, and
`call.hangup`. It is also the `connection_id` used by the outbound Dial API. A
forwarding-only phone-number connection is not enough for media streaming through
this plugin.

With `--setup-telnyx`, the examples create a temporary Call Control App and
delete it on normal shutdown. The inbound example also routes the Telnyx number
to the temporary app and restores the previous routing on shutdown.

Without `--setup-telnyx`, the examples validate the common setup requirements:

- `TELNYX_CALL_CONTROL_APP_ID` exists and is active
- the Call Control App webhook URL matches `https://<NGROK_URL>/telnyx/events`
- inbound phone numbers are routed to the Call Control App
- restricted accounts verify outbound destination numbers before dialing

## Components

### TelnyxCall

Dataclass representing an active call session:

```python
@dataclass
class TelnyxCall:
    call_control_id: str
    token: str
    webhook_data: Optional[dict[str, Any]]
    telnyx_stream: Optional[TelnyxMediaStream]
    stream_call: Optional[Any]
    started_at: datetime
    ended_at: Optional[datetime]

    # Convenience properties from Telnyx webhook payloads
    from_number: Optional[str]
    to_number: Optional[str]
    call_status: Optional[str]
```

### TelnyxCallRegistry

In-memory registry for managing active calls:

```python
registry = telnyx.CallRegistry()
registry.create(call_control_id, webhook_data=webhook_data)  # Register new call
registry.get(call_control_id)                                # Look up call
registry.require(call_control_id)                            # Look up or raise
registry.validate(call_control_id, token)                    # Validate media URL token
registry.remove(call_control_id)                             # Remove and mark ended
registry.list_active()                                       # List active calls
```

### TelnyxMediaStream

Manages Telnyx Media Streaming WebSocket connections:

```python
stream = telnyx.MediaStream(websocket)
await stream.accept()

# Access the audio track for publishing
stream.audio_track  # AudioStreamTrack matching the Telnyx media format

# Send audio back to Telnyx when bidirectional RTP streaming is enabled
await stream.send_audio(pcm_data)

# Run until the stream ends
await stream.run()
```

To send audio back to the call, start Telnyx streaming with
`stream_bidirectional_mode=rtp`. The plugin supports PCMU and PCMA at 8 kHz, and
L16 at 16 kHz.

Use `attach_phone_to_call` to bridge audio between a Telnyx media stream and a
Stream call:

```python
await telnyx.attach_phone_to_call(stream_call, stream, user_id="phone-user")
```

## Audio Utilities

```python
from vision_agents.plugins.telnyx import (
    TELNYX_DEFAULT_SAMPLE_RATE,
    TELNYX_L16_SAMPLE_RATE,
    l16_to_pcm,
    pcma_to_pcm,
    pcm_to_l16,
    pcm_to_pcma,
    pcm_to_pcmu,
    pcm_to_telnyx_payload,
    pcmu_to_pcm,
    telnyx_payload_to_pcm,
)

pcm = pcmu_to_pcm(payload)
payload = pcm_to_pcmu(pcm)
```

## Configuration

| Parameter                    | Description                          | Default |
|------------------------------|--------------------------------------|---------|
| `TELNYX_DEFAULT_SAMPLE_RATE` | Telnyx PCMU and PCMA sample rate     | `8000`  |
| `TELNYX_L16_SAMPLE_RATE`     | Telnyx L16 bidirectional sample rate | `16000` |

## Environment Variables

- `TELNYX_API_KEY`: Your Telnyx API key for Call Control API requests.
- `TELNYX_PHONE_NUMBER`: Telnyx caller ID or inbound number, in E.164 format.
  You can also pass this as `--from` or `--phone-number`.
- `NGROK_URL`: Public HTTPS hostname that forwards to your local example server.
  The examples can also auto-detect a local ngrok tunnel.
- `TELNYX_CALL_CONTROL_APP_ID`: Existing Call Control App ID. Required only when
  running without `--setup-telnyx`.
- `TELNYX_PHONE_NUMBER_ID`: Telnyx phone number resource ID. Required for
  inbound only when running without `--setup-telnyx`.

## Dependencies

- vision-agents
- numpy
- fastapi
