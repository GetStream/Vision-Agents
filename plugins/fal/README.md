# FAL.ai Wizper Speech-to-Text and Translation Plugin

High-quality real-time Speech-to-Text (STT) **and translation** for Stream, powered by
[FAL.ai Wizper](https://fal.ai/models/wizper).

## Installation

```bash
pip install getstream-plugins-fal
```

## Quick start

```python
from getstream.plugins.fal import FalWizperSTT

# 1. Pure transcription (default)
stt = FalWizperSTT()

# 2. Translation to Spanish ("es")
stt = FalWizperSTT(target_language="es")

@stt.on("transcript")
async def on_transcript(text: str, user: dict, metadata: dict):
    print(f"{user['name']} said → {text}")

# Send Stream PCM audio frames to the plugin
await stt.process_audio(pcm_data)

# Close when finished
await stt.close()
```

The plugin emits the standard Stream STT events:

* `transcript` – final, high-confidence text
* `error` – if Wizper returns a failure

## Configuration options

| Parameter          | Type   | Default     | Description                                  |
|--------------------|--------|-------------|----------------------------------------------|
| `target_language`  | str\|None | `None`      | ISO-639-1 code used when `task="translate"`   |
| `sample_rate`      | int    | `48000`     | Incoming PCM sample rate (Hz)                |

## Requirements

* Python 3.10+

## Why Wizper?

Wizper is FAL.ai’s hosted version of Whisper v3 that streams results in
real-time. This means you get:

* Accurate multilingual transcription out-of-the-box
* Fast first-word latency suitable for live calls
* Optional on-the-fly translation to 100+ languages
