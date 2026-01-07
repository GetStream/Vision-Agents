# Vision Agents - NVIDIA Nemotron Speech Plugin

NVIDIA Nemotron Speech STT integration for Vision Agents.

## Installation

```bash
pip install vision-agents-plugins-nemotron
```

## Quick Start

### 1. Start the Nemotron Server

```bash
cd plugins/nemotron/server

# Option A: Direct Python (requires NeMo)
pip install -r requirements.txt
python nemotron_server.py

# Option B: Docker
docker build -t nemotron-server .
docker run -p 8765:8765 nemotron-server
```

### 2. Use the Plugin

```python
from vision_agents.plugins import nemotron

stt = nemotron.STT(server_url="http://localhost:8765")
await stt.start()

await stt.process_audio(pcm_data)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server_url` | `http://localhost:8765` | Nemotron server URL |
| `timeout` | `30.0` | HTTP request timeout (seconds) |

## Server Configuration

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMOTRON_DEVICE` | `cpu` | Device: cpu or cuda |
| `NEMOTRON_MODEL` | `nvidia/nemotron-speech-streaming-en-0.6b` | HuggingFace model |

## Links

- [Nemotron Speech on HuggingFace](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
