# Grok TTS Plugin

A Text-to-Speech (TTS) plugin for [Vision Agents](https://github.com/GetStream/Vision-Agents) powered by [xAI's Grok Voice API](https://x.ai/api/voice). Provides five expressive voices with inline speech tags for fine-grained delivery control.

## Features

- Five distinct voices: Eve, Ara, Leo, Rex, Sal
- Inline speech tags for expressive delivery (`[laugh]`, `[pause]`, `<whisper>`, etc.)
- Multiple output codecs: PCM, MP3, WAV, mu-law, A-law
- Configurable sample rate (8 kHz – 48 kHz)
- 20+ supported languages with automatic detection
- Built-in retry with exponential backoff
- Async HTTP via aiohttp for non-blocking synthesis

## Installation

```bash
uv add "vision-agents[grok-tts]"
# or directly
uv add vision-agents-plugins-grok-tts
```

## Usage

```python
from vision_agents.plugins import grok_tts

# Default voice (eve) — energetic, upbeat
tts = grok_tts.TTS()

# Specify a voice
tts = grok_tts.TTS(voice="ara")   # warm, friendly
tts = grok_tts.TTS(voice="leo")   # authoritative, strong
tts = grok_tts.TTS(voice="rex")   # confident, clear
tts = grok_tts.TTS(voice="sal")   # smooth, balanced

# Custom output format
tts = grok_tts.TTS(
    voice="rex",
    codec="mp3",
    sample_rate=44100,
    bit_rate=192000,
)

# Explicit API key (otherwise reads XAI_API_KEY env var)
tts = grok_tts.TTS(api_key="xai-your-key-here")
```

## Configuration

| Parameter     | Type   | Default   | Description                                                           |
|---------------|--------|-----------|-----------------------------------------------------------------------|
| `api_key`     | str    | env var   | xAI API key. Falls back to `XAI_API_KEY` environment variable.        |
| `voice`       | str    | `"eve"`   | Voice ID: `"eve"`, `"ara"`, `"leo"`, `"rex"`, or `"sal"`.            |
| `language`    | str    | `"en"`    | BCP-47 language code or `"auto"` for detection.                       |
| `codec`       | str    | `"pcm"`   | Output codec: `"pcm"`, `"mp3"`, `"wav"`, `"mulaw"`, `"alaw"`.       |
| `sample_rate` | int    | `24000`   | Sample rate: `8000`–`48000` Hz.                                       |
| `bit_rate`    | int    | `None`    | MP3 bit rate (only used with `codec="mp3"`).                          |
| `base_url`    | str    | `None`    | Override the xAI TTS API endpoint.                                    |
| `session`     | object | `None`    | Optional pre-existing `aiohttp.ClientSession`.                        |

## Voices

| Voice | Tone                     | Best For                                      |
|-------|--------------------------|-----------------------------------------------|
| `eve` | Energetic, upbeat        | Demos, announcements, upbeat content (default) |
| `ara` | Warm, friendly           | Conversational interfaces, hospitality         |
| `leo` | Authoritative, strong    | Instructional, educational, healthcare         |
| `rex` | Confident, clear         | Business, corporate, customer support          |
| `sal` | Smooth, balanced         | Versatile — works for any context              |

## Speech Tags

Add expressiveness to synthesized speech with inline and wrapping tags:

**Inline tags** (placed where the expression should occur):
- Pauses: `[pause]` `[long-pause]` `[hum-tune]`
- Laughter: `[laugh]` `[chuckle]` `[giggle]` `[cry]`
- Mouth sounds: `[tsk]` `[tongue-click]` `[lip-smack]`
- Breathing: `[breath]` `[inhale]` `[exhale]` `[sigh]`

**Wrapping tags** (wrap text to change delivery):
- Volume: `<soft>text</soft>` `<loud>text</loud>` `<shout>text</shout>`
- Pitch/speed: `<high-pitch>text</high-pitch>` `<low-pitch>text</low-pitch>` `<slow>text</slow>` `<fast>text</fast>`
- Style: `<whisper>text</whisper>` `<sing>text</sing>`

## Supported Languages

| Language              | Code    |
|-----------------------|---------|
| English               | `en`    |
| Chinese (Simplified)  | `zh`    |
| French                | `fr`    |
| German                | `de`    |
| Spanish (Spain)       | `es-ES` |
| Spanish (Mexico)      | `es-MX` |
| Japanese              | `ja`    |
| Korean                | `ko`    |
| Portuguese (Brazil)   | `pt-BR` |
| Italian               | `it`    |
| Hindi                 | `hi`    |
| Arabic (Egypt)        | `ar-EG` |
| Russian               | `ru`    |
| Turkish               | `tr`    |
| Vietnamese            | `vi`    |
| Auto-detect           | `auto`  |

## Dependencies

- Python 3.10+
- aiohttp >= 3.9
- vision-agents (core)
- Optional: pydub (for MP3 decoding)

## Getting Your API Key

1. Go to [console.x.ai](https://console.x.ai/team/default/api-keys)
2. Create a new API key
3. Set the `XAI_API_KEY` environment variable or pass it directly to the plugin
