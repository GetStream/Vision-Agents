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
- **LLM**: Telnyx Inference via the OpenAI-compatible Chat Completions API
- **STT**: Streaming speech to text over WebSocket
- **TTS**: Streaming text to speech over WebSocket

## Installation

```bash
uv add "vision-agents[telnyx]"
# or directly
uv add vision-agents-plugins-telnyx
```

## Usage

Run a voice agent end to end on Telnyx. Telnyx STT does not emit VAD signals, so
pair it with a turn detector such as `smart_turn`.

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, smart_turn, telnyx

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Assistant", id="agent"),
    instructions="You are a helpful voice assistant.",
    stt=telnyx.STT(),
    llm=telnyx.LLM(),
    tts=telnyx.TTS(),
    turn_detection=smart_turn.TurnDetection(),
)
```

To bridge a PSTN phone call into a Stream call, use the Call Control primitives.
Your FastAPI server registers the call from a Telnyx webhook, answers with a
tokenized media URL, then bridges the media WebSocket into the Stream call:

```python
from vision_agents.plugins import telnyx

registry = telnyx.CallRegistry()

# 1. In your webhook handler, register the call and pre-warm the agent
call = registry.create(call_id, webhook_data=data, prepare=lambda: prepare_call(call_id))
stream_url = f"wss://{NGROK_URL}/telnyx/media/{call_id}/{call.token}"
# answer/dial via the Telnyx API with stream_url

# 2. In your media WebSocket handler, bridge the audio into the Stream call
call = registry.validate(call_id, token)
stream = telnyx.MediaStream(websocket)
await stream.accept()
agent, phone_user, stream_call = await call.await_prepare()
await telnyx.attach_phone_to_call(stream_call, stream, phone_user.id)
await stream.run()
```

See [examples/](examples/) for complete, runnable inbound and outbound servers.

## LLM

Telnyx Inference is OpenAI-compatible, so the LLM is a thin wrapper over
`ChatCompletionsLLM` pointed at `https://api.telnyx.com/v2/ai`. Streaming and
tool calling work the same as any other Chat Completions provider.

```python
from vision_agents.plugins import telnyx

llm = telnyx.LLM(model="openai/gpt-4o")
```

Requires `TELNYX_API_KEY` in the environment, or an `api_key` argument.

Model ids come from the Telnyx catalogue at `GET /v2/ai/models` and are not
validated locally. The default is `meta-llama/Llama-3.3-70B-Instruct`.

## STT

```python
from vision_agents.plugins import telnyx

# 8000 matches the PCMU telephony audio that TelnyxMediaStream decodes,
# so nothing is upsampled on the way to the transcriber.
stt = telnyx.STT(sample_rate=8000)
```

Requires `TELNYX_API_KEY` in the environment, or an `api_key` argument.

Audio is resampled to `sample_rate` and sent as raw `linear16` frames. Pick the
engine with `transcription_engine`; the default is `Telnyx`. The engine
catalogue is served by Telnyx and is not validated locally.

Telnyx does not send VAD signals on this endpoint, so the plugin emits
transcripts only and leaves turn detection to the agent.

`interim_results` is honoured per engine rather than per endpoint, and defaults
to `False`. Measured against the live API with the same audio, `Speechmatics`
and `Soniox` stream partial transcripts, while `Telnyx` and `Deepgram` accept
the parameter and return finals only:

```python
stt = telnyx.STT(transcription_engine="Speechmatics", interim_results=True)
```

## TTS

```python
from vision_agents.plugins import telnyx

tts = telnyx.TTS(voice="AWS.Polly.Danielle-Neural")
```

Requires `TELNYX_API_KEY` in the environment, or an `api_key` argument.

Voice ids come from `GET /v2/text-to-speech/voices`. The default is
`Telnyx.KokoroTTS.af_heart`.

Telnyx serves each synthesis on its own WebSocket and closes the socket after
the stop frame, so the plugin reconnects per `stream_audio` call. Audio arrives
as MP3 and is decoded to `PcmData` as it streams. The output sample rate follows
the voice, so it is taken from the decoder rather than configured.

The endpoint takes an `audio_format` parameter, but it is honoured only by some
voices — `AWS.Polly.*` and `Telnyx.NaturalHD.*` serve raw PCM, while the default
`Telnyx.KokoroTTS.*` returns MP3 regardless. Since the PCM sample rate is not
reported on the wire and differs per voice, the plugin decodes MP3 for every
voice rather than carrying a voice-to-rate table that would go stale.

## Examples

The fastest way to try `telnyx.STT`, `telnyx.LLM`, and `telnyx.TTS` is
[examples/voice_bot.py](examples/voice_bot.py), which joins a Stream call in your
browser and needs no phone number, ngrok, or Call Control App:

```bash
uv run plugins/telnyx/examples/voice_bot.py run
```

The phone examples in [examples/](examples/) bridge real PSTN calls and require
the full Telnyx Call Control setup:

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

# Inbound call answered by an all-Telnyx STT/LLM/TTS pipeline
uv run plugins/telnyx/examples/voice_agent_call.py \
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

### LLM

| Parameter         | Description                                                   | Default                          |
|-------------------|---------------------------------------------------------------|----------------------------------|
| `model`           | Model id as served by Telnyx Inference                        | `meta-llama/Llama-3.3-70B-Instruct` |
| `api_key`         | Telnyx API key (falls back to `TELNYX_API_KEY`)               | `None`                           |
| `base_url`        | API base URL                                                  | `https://api.telnyx.com/v2/ai`   |
| `client`          | Pre-configured `AsyncOpenAI` client (overrides key/base URL)  | `None`                           |
| `tools_max_rounds`| Max calling rounds for multi-hop tool calls                   | `3`                              |

### STT

| Parameter              | Description                                                            | Default   |
|------------------------|-----------------------------------------------------------------------|-----------|
| `api_key`              | Telnyx API key (falls back to `TELNYX_API_KEY`)                       | `None`    |
| `transcription_engine` | Engine to transcribe with, e.g. `Telnyx`, `Deepgram`, `Speechmatics`  | `Telnyx`  |
| `language`             | Language code                                                         | `en`      |
| `sample_rate`          | Rate in Hz audio is resampled to (use `8000` for telephony audio)     | `16000`   |
| `interim_results`      | Emit partial transcripts (honoured per engine)                        | `False`   |
| `model`                | Optional engine-specific model id                                     | `""`      |

### TTS

| Parameter         | Description                                                   | Default                     |
|-------------------|---------------------------------------------------------------|-----------------------------|
| `api_key`         | Telnyx API key (falls back to `TELNYX_API_KEY`)               | `None`                      |
| `voice`           | Voice id from `GET /v2/text-to-speech/voices`                 | `Telnyx.KokoroTTS.af_heart` |
| `idle_timeout`    | Seconds of server silence before synthesis is treated as done | `10.0`                      |
| `connect_timeout` | Seconds to wait for the WebSocket handshake                   | `10.0`                      |

### Audio constants

| Constant                     | Description                          | Value  |
|------------------------------|--------------------------------------|--------|
| `TELNYX_DEFAULT_SAMPLE_RATE` | Telnyx PCMU and PCMA sample rate     | `8000` |
| `TELNYX_L16_SAMPLE_RATE`     | Telnyx L16 bidirectional sample rate | `16000`|

## Environment Variables

- `TELNYX_API_KEY`: Your Telnyx API key for Call Control, Inference, STT, and TTS.
- `TELNYX_PUBLIC_KEY`: Base64 Ed25519 public key from the Telnyx Mission Control
  Portal. The phone examples verify webhook signatures before handling events.
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
- vision-agents-plugins-openai
- cryptography
- fastapi
- aiohttp
- numpy
