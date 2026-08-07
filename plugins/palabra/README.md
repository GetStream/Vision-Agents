# Palabra AI

[Palabra AI](https://palabra.ai) provides a realtime Text-to-Speech (TTS) API built for streaming: text is accepted
incrementally over a WebSocket and audio comes back as raw PCM within a few hundred milliseconds, which makes it a good
fit for voice AI agents.

The Palabra plugin for Vision Agents lets you give your agent a Palabra voice, in 25 languages and with your own cloned
voices.

## Features

- Streaming TTS over a single persistent WebSocket – the session is opened once and reused for every utterance
- Sentence-level streaming (`streaming = True`), so the agent starts speaking while the LLM is still writing
- Instant barge-in: `stop_audio()` cancels synthesis server-side without dropping the connection
- Raw PCM output at any sample rate between 8 kHz and 48 kHz
- 25 languages
- Voice cloning from an audio sample, via `palabra.Voices`

## Installation

```bash
uv add "vision-agents[palabra]"
# or directly
uv add vision-agents-plugins-palabra
```

## Usage

```python
from vision_agents.plugins import palabra

tts = palabra.TTS()
```

Use it in an agent:

```python
from vision_agents.core.agents import Agent
from vision_agents.core.edge.types import User
from vision_agents.plugins import deepgram, gemini, getstream, palabra

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Palabra Voice Bot", id="agent"),
    instructions="You're a helpful voice AI assistant.",
    stt=deepgram.STT(),
    llm=gemini.LLM(),
    tts=palabra.TTS(voice_id="default_high", language="en"),
)
```

<Warning>
  To initialise without passing in the API key, make sure `PALABRA_API_KEY` is available as an environment variable.
  You can do this either by defining it in a `.env` file or exporting it directly in your terminal. Create a key in the
  [Palabra platform](https://platform.palabra.ai/api-keys).
</Warning>

## Examples

Check out our [Palabra example](https://github.com/GetStream/Vision-Agents/tree/main/plugins/palabra/example) to see
working code:

- [main.py](https://github.com/GetStream/Vision-Agents/blob/main/plugins/palabra/example/main.py) – a voice bot that
  uses Palabra TTS in a Stream call
- [tts_smoke.py](https://github.com/GetStream/Vision-Agents/blob/main/plugins/palabra/example/tts_smoke.py) – synthesize
  a few sentences to a WAV file and report the time to first audio chunk
- [clone_voice.py](https://github.com/GetStream/Vision-Agents/blob/main/plugins/palabra/example/clone_voice.py) – clone a
  voice from an audio sample and speak with it

## Configuration

| Name                | Type              | Default           | Description                                                                                              |
|---------------------|-------------------|-------------------|----------------------------------------------------------------------------------------------------------|
| `api_key`           | `str` or `None`   | `None`            | Your Palabra API key. Falls back to the `PALABRA_API_KEY` environment variable.                          |
| `voice_id`          | `str`             | `"default_low"`   | Voice to synthesize with: `default_low`, `default_high`, or the id of a [cloned voice](https://platform.palabra.ai/docs/assets/voices). |
| `language`          | `str`             | `"en"`            | BCP-47 language code of the text, e.g. `en`, `en-gb`, `de`, `pt-eu`, `ko`.                               |
| `model`             | `str`             | `"auto"`          | TTS model id. `auto` lets Palabra pick the model.                                                        |
| `sample_rate`       | `int`             | `24000`           | Output sample rate in Hz. Must be between `8000` and `48000`.                                            |
| `speed`             | `float` or `None` | `None`            | Speech speed multiplier between `0.0` and `2.0`. `None` uses the server default.                         |
| `deaccent_strength` | `float` or `None` | `None`            | Accent reduction for cloned voices, between `0.0` and `1.0`. `None` uses the server default.             |
| `ws_url`            | `str`             | `WS_URL_EU`       | Palabra endpoint. Pass `palabra.WS_URL_US` to use the US region.                                          |
| `idle_timeout`      | `float`           | `5.0`             | Seconds to wait for the next audio frame before abandoning a generation. Guards against a server that stops answering without sending an error. |

## Functionality

### Send text to convert to speech

`send_iter()` sends the text to Palabra and yields `TTSOutputChunk`s carrying the produced PCM audio:

```python
async for chunk in tts.send_iter("Demo text you want the AI voice to say"):
    pass
```

Text longer than Palabra's 1024-character per-message limit is split across several messages automatically, on word
boundaries.

### Stop speaking

```python
await tts.stop_audio()
```

This sends a `cancel` to Palabra and drops any audio still in flight. The WebSocket session stays open, so the next
utterance does not pay for a new handshake.

## Voice cloning

`palabra.Voices` wraps Palabra's [cloned voice API](https://platform.palabra.ai/docs/assets/voices). `clone()` runs the
whole sequence — reserve the voice, upload the sample to the presigned target, poll until Palabra reports it `ready` —
and returns a `voice_id` you pass straight to `TTS`:

```python
from vision_agents.plugins import palabra

async with palabra.Voices() as voices:
    voice = await voices.clone("Narrator", "sample.wav", lang_code="en")

tts = palabra.TTS(voice_id=voice.voice_id, deaccent_strength=0.7)
```

`deaccent_strength` exists specifically for cloned voices: lower it to keep more of the speaker's accent, raise it to
neutralise it.

Palabra needs **at least 30 seconds** of clean, single-speaker audio, at most **10 MB**, as MP3, WAV, FLAC, WEBM, MP4,
MPEG or MPG. Cloning usually finishes in under a minute; `clone()` polls until then, or pass `wait=False` to return
immediately and check `voices.get(voice_id).ready` yourself.

| Method                                    | Description                                                                       |
|-------------------------------------------|-----------------------------------------------------------------------------------|
| `clone(name, sample, ...)`                | Clone a voice from an audio or video sample. Returns a `ClonedVoice`.              |
| `get(voice_id)`                           | Fetch one voice, including `processing_status` and any errors or warnings.         |
| `list(search=..., lang=..., page_size=...)`| List cloned voices.                                                              |
| `delete(voice_id)`                        | Permanently delete a cloned voice. Irreversible.                                  |
| `limits()`                                | Cloned voice quota for the account (`total`, `limit`, `remaining`, …).             |

Cloned voices count against your account quota, so delete the ones you no longer need. Only clone a person's voice with
their explicit consent.

## Dependencies

- [`vision-agents`](https://pypi.org/project/vision-agents/)
- [`websockets`](https://pypi.org/project/websockets/)
- [`httpx`](https://pypi.org/project/httpx/) – for the voice cloning REST API
