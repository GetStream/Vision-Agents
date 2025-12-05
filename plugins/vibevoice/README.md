# Vision Agents VibeVoice Plugin

VibeVoice TTS plugin for Vision Agents, supporting both local execution and remote WebSocket connections.

## Features

- **Local Execution**: Downloads and runs Microsoft VibeVoice-Realtime-0.5B locally (requires GPU for best performance).
- **Remote Streaming**: Connects to a VibeVoice WebSocket server for remote inference.
- **Voice Presets**: Automatically manages and downloads voice preset files.
- **Streaming Audio**: Returns audio chunks as they are generated for low-latency playback.

## Installation

```bash
uv add vision-agents-plugins-vibevoice
```

## Usage

```python
from vision_agents.plugins import vibevoice

# 1. Local Execution (default)
tts_local = vibevoice.TTS()
async for chunk in tts_local.stream_audio("Hello world from local model"):
    pass

# 2. Remote Execution
tts_remote = vibevoice.TTS(base_url="ws://localhost:3000/stream")
async for chunk in tts_remote.stream_audio("Hello world from remote server"):
    pass
```

## Configuration

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `base_url` | WebSocket URL for remote mode. If None, runs locally. | `None` (env: `VIBEVOICE_URL`) |
| `model_id` | HuggingFace model ID for local execution. | `microsoft/VibeVoice-Realtime-0.5B` |
| `device` | Device to run on (`cuda`, `cpu`, `mps`). | Auto-detect (env: `VIBEVOICE_DEVICE`) |
| `voice` | Voice preset name. | `en-WHTest_man` |
| `cfg_scale` | Guidance scale for generation. | `1.5` |
| `inference_steps` | Number of diffusion steps. | `5` |

## Dependencies

- `vision-agents`
- `vibevoice` (git)
- `huggingface_hub`
- `torch`
- `numpy`
- `websockets`
