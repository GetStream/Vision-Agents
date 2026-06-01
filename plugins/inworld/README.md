# Inworld AI Plugin

Inworld AI integration for Vision Agents. Provides:

- **LLM / VLM** — text and vision chat completions through the Inworld
  Realtime Router, which proxies upstream across OpenAI / Anthropic / Google
  / etc. with auto-selection, fallbacks, and traffic splitting.
- **TTS** — high-quality streaming text-to-speech.
- **Realtime** — WebRTC speech-to-speech for low-latency voice agents.

## Installation

```bash
uv add "vision-agents[inworld]"
# or directly
uv add vision-agents-plugins-inworld
```

Get your API key from the [Inworld Portal](https://studio.inworld.ai/) and set
`INWORLD_API_KEY` in your environment (or pass `api_key=` explicitly).

## LLM / VLM (router)

`inworld.LLM` and `inworld.VLM` hit Inworld's OpenAI-compatible
`/v1/chat/completions` endpoint. The `model` argument accepts:

- `"inworld/<router-id>"` — a router defined in the Inworld portal
- `"<provider>/<model-id>"` — e.g. `"openai/gpt-4o-mini"`
- `"auto"` — let Inworld pick (combine with `sort_by`)

```python
from vision_agents.plugins import inworld

# Lowest-latency routing with a small fallback chain
llm = inworld.LLM(
    model="auto",
    sort_by=["latency"],
    ttft_timeout="500ms",
    fallback_models=["openai/gpt-4o-mini", "google-ai-studio/gemini-2.5-flash"],
)

# Vision over the router (frames sent as image_url content).
# Tuned for low-latency video Q&A: small frames, short buffer, fast fallback.
vlm = inworld.VLM(
    model="auto",
    sort_by=["latency"],
    ttft_timeout="500ms",
    fallback_models=["google-ai-studio/gemini-2.5-flash", "openai/gpt-4o-mini"],
    fps=1,
    frame_buffer_seconds=3,
    frame_width=512,
    frame_height=384,
)
```

See `example/inworld_llm_example.py` and `example/inworld_vlm_example.py`
for end-to-end voice and video agents respectively.

### Tuning the VLM for latency

Video Q&A latency is dominated by **input tokens** (frames cost a lot more
than text) and **upstream choice**. The example values above are the right
starting point:

- `frame_width=512, frame_height=384` — 4× fewer bytes than the 800×600
  default, with negligible accuracy loss for typical Q&A.
- `frame_buffer_seconds=3` with `fps=1` — 3 frames/request. Longer buffers
  inflate input tokens quadratically without helping short-horizon questions.
- `sort_by=["latency"]` + `ttft_timeout="500ms"` + a fallback chain to
  fast vision models keeps TTFT predictable when one provider degrades.

### When to use which

- **Voice agents** → `inworld.Realtime` (WebRTC, full-duplex, lowest latency).
  The text router cannot beat full-duplex audio for STT→LLM→TTS pipelines.
- **Text agents, STT→LLM→TTS pipelines, video Q&A** → `inworld.LLM` / `inworld.VLM`.

### Routing kwargs

- `fallback_models`: ordered list, tried on failure.
- `ignore_models`: excluded from `auto`.
- `sort_by`: any of `"price"`, `"latency"`, `"throughput"`, `"intelligence"`, `"math"`, `"coding"`. Multiple metrics rank with tiebreakers.
- `ttft_timeout`: switch to fallback if first token doesn't arrive in time (Inworld minimum `"300ms"`).
- `metadata`: free-form dict consumed by router CEL expressions for conditional routing.
- `web_search` / `web_search_options`: opt-in upstream web grounding.
- `compression_aggressiveness` (0–1): Inworld's prompt compression applied to the system message — cuts input tokens, lowers TTFT for long prompts.
- `extra_body`: raw escape hatch merged in last.

### Caching

Implicit prompt caching is automatic on OpenAI / DeepSeek / Gemini-2.5
upstreams — no code needed. Explicit caching (Anthropic / Google) is a
per-message thing: add a `cache_control` block to the message content
yourself, e.g. `{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}`.

Router definitions themselves (router IDs, A/B variants, traffic weights)
are configured in the Inworld portal — out of scope for this plugin.

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

A complete example wiring `inworld.TTS()` into a Stream-edge agent with
Deepgram STT, Gemini LLM, and smart-turn detection lives at
[`example/inworld_tts_example.py`](example/inworld_tts_example.py). The
companion [`example/inworld-audio-guide.md`](example/inworld-audio-guide.md)
is loaded as the agent's system prompt and teaches the LLM how to emit
TTS-2 steering tags so replies sound expressive out of the box.

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
