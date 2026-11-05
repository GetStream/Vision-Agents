# Audexum

[Audexum](https://audexum.com) is a text-to-speech and speech-to-text API. It has a free tier (2,000 credits to start, then 250 per day) and paid plans.

The Audexum plugin for the Stream Python AI SDK adds TTS to your project. It calls Audexum's OpenAI-compatible `/v1/audio/speech` endpoint and returns PCM audio.

## Installation

```bash
uv add "vision-agents[audexum]"
# or directly
uv add vision-agents-plugins-audexum
```

## Initialisation

```python
from vision_agents.plugins import audexum

tts = audexum.TTS()
```

Create an API key at [audexum.com](https://audexum.com) and export `AUDEXUM_API_KEY` in your shell, or put it in a `.env` file and load it (for example with `python-dotenv`'s `load_dotenv()`) before creating `audexum.TTS()`.

## Parameters

| Name       | Type            | Default  | Description                                                                                     |
|------------|-----------------|----------|-------------------------------------------------------------------------------------------------|
| `api_key`  | `str` or `None` | `None`   | Your Audexum API key. Falls back to the `AUDEXUM_API_KEY` environment variable.                 |
| `voice`    | `str`           | `"Nova"` | Voice name, e.g. `Nova`, `Atlas`, `Reid`, `F1`, `M1`. List: `GET https://audexum.com/v1/voices`. |
| `speed`    | `float`         | `1.0`    | Speaking-rate multiplier, 0.25 to 4.0.                                                          |
| `base_url` | `str` or `None` | `None`   | Override the API base URL (default `https://audexum.com/v1`).                                   |

## Functionality

The API returns a complete clip per request (no incremental streaming), so latency grows with text length. Send short sentences for conversational use.

```python
async for chunk in tts.send_iter("Demo text you want AI voice to say"):
    pass
```
