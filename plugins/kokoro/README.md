# Kokoro TTS Plugin

This package integrates the open-weight [Kokoro-82M TTS model](https://github.com/hexgrad/kokoro) with Vision Agents.

Kokoro runs locally and produces 24 kHz mono PCM audio. Model initialization is handled by the Vision Agents warmup lifecycle.

```python
from vision_agents.plugins import kokoro

tts = kokoro.TTS(lang_code="a", voice="af_heart")

try:
    await tts.warmup()
    audio_chunks = []
    async for chunk in tts.send_iter("Hello from Kokoro!"):
        if chunk.data:
            audio_chunks.append(chunk.data)
finally:
    await tts.close()
```

## Installation

```bash
uv add "vision-agents[kokoro]"
# or directly
uv add vision-agents-plugins-kokoro
```

This installs the required `kokoro`, `misaki`, and `numpy` dependencies. You also need `espeak-ng` at runtime for phonemization. On macOS you can install it with Homebrew:

```bash
brew install espeak-ng
```

## Configuration options

| Parameter     | Default      | Description                                                                                                                      |
|---------------|--------------|----------------------------------------------------------------------------------------------------------------------------------|
| `lang_code`   | `"a"`        | Language group passed to `KPipeline` (`"a"` is American English).                                                               |
| `voice`       | `"af_heart"` | Kokoro voice preset. See the [Kokoro voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).                       |
| `speed`       | `1.0`        | Playback speed multiplier.                                                                                                       |
| `sample_rate` | `24000`      | Output sample rate. Kokoro produces 24 kHz audio.                                                                                |
| `device`      | `None`       | Optional device passed to `KPipeline`, such as `"cpu"` or `"cuda"`.                                                            |
| `client`      | `None`       | Optional pre-initialized `KPipeline`.                                                                                            |
