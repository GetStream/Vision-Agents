# Voice.ai
"Voice.ai"

## Features
- Low-latency HTTP streaming TTS via Voice.ai.
- PCM output ready for Vision Agents audio pipeline.
- Configurable voice, model, language, temperature, and top-p.

## Installation
```sh
uv add vision-agents-plugins-voice-ai
```

## Usage
```python
from vision_agents.plugins import voice_ai

# Requires VOICE_AI_API_KEY and VOICE_AI_VOICE_ID in the environment
tts = voice_ai.TTS()
```

## Configuration
| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `api_key` | `str` or `None` | `None` | Voice.ai API key. Falls back to `VOICE_AI_API_KEY`. |
| `voice_id` | `str` or `None` | `None` | Voice ID to use. Falls back to `VOICE_AI_VOICE_ID`. |
| `audio_format` | `str` | `"pcm"` | Output format. `pcm` streams and yields chunks; `wav`/`mp3` are decoded after download. |
| `model` | `str` or `None` | `None` | Model ID to use for synthesis. |
| `language` | `str` or `None` | `None` | Language code for synthesis. |
| `temperature` | `float` or `None` | `None` | Sampling temperature. |
| `top_p` | `float` or `None` | `None` | Top-p nucleus sampling. |
| `base_url` | `str` | `"https://dev.voice.ai"` | API base URL. |
| `timeout_s` | `float` | `60.0` | HTTP timeout in seconds. |
| `client` | `httpx.AsyncClient` or `None` | `None` | Optional pre-configured HTTP client. |

## Voice IDs
Use the Voice.ai voices list endpoint to discover `voice_id` values:
- `GET /api/v1/tts/voice/list`

## Dependencies
- `vision-agents`
- `httpx`
- `av`
