# Vision Agents — FunASR STT plugin

Local, self-hosted speech-to-text for [Vision Agents](https://visionagents.ai/),
powered by [FunASR](https://github.com/modelscope/FunASR) (SenseVoice / Fun-ASR-Nano /
Paraformer). Strong on Chinese, Cantonese, Japanese, Korean and more; runs locally on
CPU or CUDA with **no API key**.

## Install

```bash
uv add vision-agents-plugins-funasr
# or: pip install vision-agents-plugins-funasr
```

## Usage

```python
from vision_agents.plugins import funasr

# Default: SenseVoice-Small (fast, multilingual, CPU-friendly)
stt = funasr.STT(model="iic/SenseVoiceSmall", language="auto", device="cpu")

# On a GPU, use the flagship LLM-ASR model:
# stt = funasr.STT(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda")
```

| Arg | Default | Description |
|---|---|---|
| `model` | `iic/SenseVoiceSmall` | FunASR model id. Use `FunAudioLLM/Fun-ASR-Nano-2512` (flagship) on GPU, or `paraformer-zh` for Chinese. |
| `language` | `auto` | Language hint, or `auto` to detect. |
| `device` | `cpu` | `cpu` or `cuda`. |
| `use_itn` | `true` | Apply inverse text normalization. |

The model downloads automatically on first run.
