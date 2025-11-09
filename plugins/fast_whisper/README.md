# Fast Whisper STT Plugin

Fast Whisper STT plugin for Vision Agents, providing real-time audio transcription using [faster-whisper](https://github.com/guillaumekln/faster-whisper).

## Features

- Fast inference using CTranslate2-based Whisper implementation
- Support for multiple model sizes (tiny, base, small, medium, large, large-v2, large-v3)
- Automatic language detection or manual language specification
- CPU and GPU support
- Quantization support (int8, float16, float32)

## Installation

```bash
uv add vision-agents[fast-whisper]
```
