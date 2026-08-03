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
- 25 languages plus cloned voices

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

## Dependencies

- [`vision-agents`](https://pypi.org/project/vision-agents/)
- [`websockets`](https://pypi.org/project/websockets/)
