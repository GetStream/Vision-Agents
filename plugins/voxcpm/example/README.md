# ModelBest VoxCPM Smoke Test

This example makes one real request to the hosted ModelBest VoxCPM API and writes the returned speech to a WAV file. It is the smallest way to verify API credentials, model access, streamed audio parsing, and optional speaker cloning before adding VoxCPM to a full Agent.

## Setup

```bash
cd plugins/voxcpm/example
cp .env.example .env
```

Fill in `MODELBEST_API_KEY` and a ModelBest model ID with the `speech_synthesis` capability. Set `VOXCPM_REFERENCE_AUDIO` to a PCM16 WAV path to test speaker cloning.

## Run

```bash
uv run voxcpm_smoke.py
uv run voxcpm_smoke.py "你好，这是 VoxCPM 的测试语音。"
```

The output is written to `voxcpm_smoke.wav`. The command prints the number of streamed chunks, sample rate, first-chunk latency, and total duration.
