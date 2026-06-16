# FunASR STT Plugin

FunASR STT plugin for Vision Agents, providing self-hosted speech-to-text using [FunASR](https://github.com/modelscope/FunASR) and its SenseVoice model.

## Features

- Self-hosted, no external API dependency or per-minute costs
- Non-autoregressive SenseVoice model (single forward pass) for low latency
- 50+ languages with automatic language detection
- Emotion detection surfaced on each transcript (`response.other["emotion"]`)
- CPU and GPU support

## Installation

```bash
uv add vision-agents[funasr]
# or directly
uv add vision-agents-plugins-funasr
```

## Usage

```python
from vision_agents.plugins import funasr

stt = funasr.STT(
    model="iic/SenseVoiceSmall",
    language="auto",  # or "en", "zh", "yue", "ja", "ko"
    device="cpu",     # use "cuda" for GPU
)
```
