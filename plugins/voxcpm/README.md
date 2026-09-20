# VoxCPM Plugin

This package integrates the hosted [ModelBest VoxCPM](https://platform.modelbest.cn/console/docs/api/audio) Text-to-Speech API with Vision Agents. It streams audio as it is synthesized and supports optional voice cloning without requiring a local GPU.

## Installation

```bash
uv add "vision-agents[voxcpm]"
# or directly
uv add vision-agents-plugins-voxcpm
```

## Usage

Create a ModelBest API key and select a model with the `speech_synthesis` capability, then set:

```bash
export MODELBEST_API_KEY="your-api-key"
export MODELBEST_VOXCPM_MODEL_ID="your-model-id"
```

```python
from vision_agents.plugins import voxcpm

tts = voxcpm.TTS()

try:
    async for chunk in tts.send_iter("Hello from VoxCPM!"):
        if chunk.data:
            print(chunk.data.duration_ms)
finally:
    await tts.close()
```

## Voice cloning

`ref_audio` sets speaker identity. `prompt_audio` and its exact `prompt_text` can additionally preserve the reference delivery, including pacing, emotion, and pronunciation.

```python
tts = voxcpm.TTS(
    ref_audio="speaker.wav",
    prompt_audio="delivery.wav",
    prompt_text="The exact words spoken in delivery.wav.",
)
```

Reference files must be valid, uncompressed WAV files no larger than 5 MiB. Audio supplied for cloning is sent to ModelBest; only use recordings you have permission to process and clone.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `MODELBEST_API_KEY` | ModelBest API key. |
| `model` | `MODELBEST_VOXCPM_MODEL_ID` | Model ID with `speech_synthesis` capability. |
| `voice` | `"default"` | Protocol voice value; cloned identity comes from `ref_audio`. |
| `base_url` | `https://api.modelbest.cn/v1` | ModelBest API base URL. |
| `ref_audio` | `None` | WAV bytes or path used for speaker identity. |
| `prompt_audio` | `None` | WAV bytes or path used for delivery cloning. |
| `prompt_text` | `None` | Exact transcript paired with `prompt_audio`. |
| `request_timeout` | `120.0` | Maximum seconds without response data. |
