# Inworld AI Plugin

Inworld AI integration for Vision Agents. Provides speech-to-text (STT),
text-to-speech (TTS), and a WebRTC-based Realtime speech-to-speech
conversational API.

## Installation

```bash
uv add "vision-agents[inworld]"
# or directly
uv add vision-agents-plugins-inworld
```

Get your API key from the [Inworld Portal](https://studio.inworld.ai/) and set
`INWORLD_API_KEY` in your environment (or pass `api_key=` explicitly).

## STT

Real-time speech-to-text via Inworld's bidirectional WebSocket streaming API.
Supports multiple STT backends, built-in turn detection, and voice activity
detection.

```python
from vision_agents.plugins import inworld

# Defaults to model_id="inworld/inworld-stt-1", language="en-US"
stt = inworld.STT()

# Or configure explicitly
stt = inworld.STT(
    model_id="inworld/inworld-stt-1",
    language="en-US",
    vad_threshold=0.6,
    min_end_of_turn_silence_when_confident=300,
    end_of_turn_confidence_threshold=0.6,
)
```

### STT options

- `api_key`: Inworld AI API key (default: reads from `INWORLD_API_KEY`)
- `model_id`: STT model identifier. Any model supported by the Inworld STT API
  is accepted — `"inworld/inworld-stt-1"` (default), `"assemblyai/universal-streaming-multilingual"`,
  `"assemblyai/u3-rt-pro"`, `"soniox/stt-rt-v4"`, etc.
- `language`: BCP-47 language code (default: `"en-US"`)
- `sample_rate`: Audio sample rate in Hz (default: `16000`)
- `end_of_turn_confidence_threshold`: Confidence threshold for end-of-turn
  detection. Range [0.0, 1.0], higher = fewer false positives (default: `0.6`)
- `vad_threshold`: Voice activity detection sensitivity. Range [0.0, 1.0]
  (default: server default). Raise to filter out background noise.
- `min_end_of_turn_silence_when_confident`: Minimum silence in ms before
  committing an end-of-turn when confidence is high (default: server default).
  Higher values give the model more context on short utterances.

### Turn detection

Inworld STT provides server-side turn detection via `speechStarted` /
`speechStopped` events. When using `inworld.STT()`, you do not need a
separate `TurnDetector` plugin:

```python
agent = Agent(
    stt=inworld.STT(),   # turn detection built in
    tts=inworld.TTS(),
    llm=my_llm,
    # no turn_detection= needed
)
```

## TTS

High-quality text-to-speech with streaming support. The plugin now defaults
to Inworld's **TTS-2** model (currently in research preview), which adds
natural-language steering, 100+ languages (15 GA, 90+ experimental), and
high-quality instant voice cloning over the previous `inworld-tts-1.5-*`
generation.

```python
from vision_agents.plugins import inworld

# Defaults to model_id="inworld-tts-2", voice_id="Sarah"
tts = inworld.TTS()

# Or specify explicitly
tts = inworld.TTS(
    api_key="your_inworld_api_key",
    voice_id="Ashley",
    model_id="inworld-tts-2",
    temperature=1.1,
)
```

### TTS options

- `api_key`: Inworld AI API key (default: reads from `INWORLD_API_KEY`)
- `voice_id`: Voice to use (default: `"Sarah"`; `"Dennis"`, `"Ashley"`, `"Olivia"`, `"Clive"` and custom/cloned voices also supported)
- `model_id`: `"inworld-tts-2"` (default), `"inworld-tts-1.5-max"`, `"inworld-tts-1.5-mini"`. `"inworld-tts-1"` and `"inworld-tts-1-max"` are deprecated by Inworld — migrate to `inworld-tts-2` or `inworld-tts-1.5-*`.
- `temperature`: 0–2 (default: 1.1)

The plugin requests `LINEAR16` (16-bit PCM WAV) chunks from Inworld so each
streamed chunk is self-contained and decodes cleanly under streaming TTS;
no extra configuration needed.

### Steering (TTS-2)

TTS-2 takes natural-language stage directions inline with your text. Place
the instruction in square brackets before the segment it should apply to:

```python
text = (
    "[whisper in a hushed style] I have to tell you something. "
    "[laugh] Just kidding! [say with force] Now let's get to work."
)
async for chunk in await tts.stream_audio(text):
    ...
```

Steering covers articulation, intonation, volume, pitch, range, speed, and
vocal style — and supports non-verbal sounds like `[laugh]`, `[breathe]`,
`[clear throat]`, `[sigh]`, `[cough]`, `[yawn]`. Combining dimensions
(`[whisper in a hushed style]`, `[say playfully and very fast]`) produces
better results than bare single-word tags. See Inworld's
[steering docs](https://docs.inworld.ai/tts/capabilities/steering) and
[prompting guide](https://docs.inworld.ai/tts/best-practices/prompting-for-tts-2)
for the full reference.

### Agent example

A complete example wiring `inworld.STT()` and `inworld.TTS()` into a
Stream-edge agent with Gemini LLM lives at
[`example/inworld_tts_example.py`](example/inworld_tts_example.py).

## Realtime (WebRTC)

Low-latency speech-to-speech via Inworld's Realtime API. This transport uses
WebRTC (UDP, native Opus) for lower latency than the WebSocket alternative.
Requires a WebRTC-capable edge transport — pair with `getstream.Edge()` as
shown below.

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, inworld, smart_turn

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="My Agent", id="agent"),
    llm=inworld.Realtime(
        model="openai/gpt-4o-mini",
        voice="Dennis",
        instructions="You are a friendly voice assistant.",
    ),
    turn_detection=smart_turn.TurnDetection(),
)
```

### Realtime options

- `model`: provider-prefixed model ID. Examples: `"openai/gpt-4o-mini"` (default), `"google-ai-studio/gemini-2.5-flash"`, `"inworld/<router-id>"` for an Inworld router
- `voice`: voice for audio responses (default: `"Dennis"`; `"Clive"`, `"Olivia"` and custom voices also supported)
- `api_key`: Inworld AI API key (default: reads from `INWORLD_API_KEY`)
- `instructions`: system prompt
- `realtime_session`: advanced — pass a full `RealtimeSessionCreateRequestParam` for session fields not exposed by the primary args (custom turn-detection, `tool_choice`, etc.)

### Registering tools

```python
realtime = inworld.Realtime()

@realtime.register_function(description="Get the current weather for a city.")
async def get_weather(city: str) -> str:
    return f"It's sunny in {city}."
```

Tools follow the OpenAI function-calling schema. Inworld's Realtime API is
protocol-compatible with OpenAI's Realtime API, so registered functions flow
through the same `response.function_call_arguments.done` path.

### Notes

- v1 is WebRTC only; a WebSocket transport may be added later.
- Video input is not currently supported by Inworld's Realtime API.

## Requirements

- Python 3.10+
- `httpx>=0.28`, `av>=10`, `aiortc>=1.9`, `openai[realtime]>=2.26,<3`
