# 60db TTS plugin

[60db](https://docs.60db.ai) provides authenticated text-to-speech using voices in a workspace. This plugin implements the Vision Agents TTS interface with complete-utterance synthesis.

## Installation

```sh
uv add 'vision-agents[sixtydb]'
```

## Usage

Set `SIXTYDB_API_KEY` and `SIXTYDB_VOICE_ID` in your process environment. Retrieve a voice ID from your workspace's `GET /voices` API; voice IDs are not shared defaults.

```python
from vision_agents.plugins import sixtydb

# Pass this instance as the Agent's tts argument.
tts = sixtydb.TTS()
# Or configure explicitly:
tts = sixtydb.TTS(api_key="your-key", voice_id="your-voice-id", speed=1.0)
```

The standalone example synthesizes a sentence through `send_iter()` and writes a WAV file:

```sh
uv run --package vision-agents-plugins-sixtydb python plugins/sixtydb/example/sixtydb_tts_example.py
```

## Configuration

| Parameter | Default | Purpose |
| --- | --- | --- |
| `api_key` | `SIXTYDB_API_KEY` | Workspace API credential |
| `voice_id` | `SIXTYDB_VOICE_ID` | Workspace voice ID |
| `model` | Service default | Optional model ID passed as `model_id` |
| `speed` | `1.0` | Speaking rate from `0.5` to `2.0` |
| `timeout` | `60.0` | Socket timeout in seconds; also checked between response reads |

The plugin requests mono LINEAR16 at 24 kHz from `POST /tts-synthesize`, buffers binary PCM/WAV or JSON/NDJSON audio envelopes, and returns `PcmData`. Responses are limited to 32 MiB. HTTP redirects are rejected to prevent forwarding credentials.

`streaming` is false: partial text deltas are not accepted. Interrupting discards stale output through the core TTS epoch; it does not cancel synthesis at the service. In-flight requests finish or time out. Requests accept 1–5000 characters. No retries are made automatically.

## Dependencies and testing

The plugin depends on `vision-agents` and uses Python's standard library for HTTP. Tests exercise a local HTTP server and the real core audio classes without service credentials. They do not verify live service synthesis or voice quality.
