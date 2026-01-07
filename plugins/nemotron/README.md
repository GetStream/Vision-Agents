# Vision Agents - NVIDIA Nemotron Speech Plugin

NVIDIA Nemotron Speech STT integration for Vision Agents.

## Installation

```bash
pip install vision-agents-plugins-nemotron
```

## Usage

```python
from vision_agents.plugins import nemotron

stt = nemotron.STT()
await stt.warmup()

await stt.process_audio(pcm_data)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `nvidia/nemotron-speech-streaming-en-0.6b` | HuggingFace model name |
| `chunk_size` | `560ms` | Processing chunk size (80ms, 160ms, 560ms, 1120ms) |
| `device` | `cpu` | Device to run on (cpu or cuda) |

## Links

- [Nemotron Speech on HuggingFace](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
