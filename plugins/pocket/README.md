# Pocket TTS Plugin

A lightweight Text-to-Speech (TTS) plugin for [Vision Agents](https://github.com/GetStream/Vision-Agents) powered by Kyutai's [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) model. Runs efficiently on CPU with low latency (~200ms) and supports voice cloning.

## Features

- Runs on CPU - no GPU required
- Small model size (100M parameters)
- Low latency (~200ms to first audio)
- Voice cloning support
- Built-in voice selection

## Installation

```bash
uv add vision-agents[pocket]
```

## Usage

```python
from vision_agents.plugins import pocket

# Create TTS with default voice
tts = pocket.TTS()

# Or specify a built-in voice
tts = pocket.TTS(voice="marius")

# Or use a custom voice for cloning
tts = pocket.TTS(voice="path/to/your/voice.wav")
```

## Configuration

| Parameter | Description | Values |
|-----------|-------------|--------|
| `voice` | Built-in voice name or path to custom wav file | `"alba"` (default), `"marius"`, `"javert"`, `"jean"`, `"fantine"`, `"cosette"`, `"eponine"`, `"azelma"`, or custom path |

## Built-in Voices

- `alba` - Default voice
- `marius`
- `javert`
- `jean`
- `fantine`
- `cosette`
- `eponine`
- `azelma`

## Dependencies

- pocket-tts>=0.1.0
- PyTorch 2.5+
