# vision-agents-plugins-modulate

[Modulate AI](https://modulate.ai/) Velma-2 streaming Speech-to-Text plugin for [Vision Agents](https://github.com/GetStream/Vision-Agents).

## Installation

```bash
uv add vision-agents["modulate"]
```

## Usage

```python
from vision_agents.plugins import modulate

stt = modulate.STT(api_key="your-modulate-api-key")
```

Set the `MODULATE_API_KEY` environment variable to avoid passing the key explicitly.

## Features

- Real-time streaming transcription via WebSocket
- Speaker diarization (enabled by default)
- Optional partial/interim results
- Optional emotion, accent, and deepfake signals
- PII/PHI tagging support
