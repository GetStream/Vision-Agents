# AssemblyAI Plugin

Streaming Speech-to-Text (STT) plugin for Vision Agents using AssemblyAI's Universal-3 Pro model.

## Features

- Real-time streaming transcription via async WebSocket
- Built-in punctuation-based turn detection with configurable silence thresholds
- Native `SpeechStarted` event support
- Custom prompt and keyterms boosting support
- Sub-300ms time to complete transcript latency
- Built-in reconnection with exponential backoff

## Installation

```bash
uv add "vision-agents[assemblyai]"
# or directly
uv add vision-agents-plugins-assemblyai
```

## Usage

```python
from vision_agents.plugins import assemblyai

stt = assemblyai.STT(
    speech_model="u3-rt-pro",  # Default model
    sample_rate=16000,
)
```

### With keyterms boosting

```python
stt = assemblyai.STT(
    keyterms_prompt=["AssemblyAI", "Vision Agents"],
)
```

### With custom turn silence thresholds

```python
stt = assemblyai.STT(
    min_turn_silence=100,   # ms before speculative EOT check
    max_turn_silence=1200,  # ms before forcing turn end
)
```

## Configuration

| Parameter                     | Description                                                               | Default       |
|-------------------------------|---------------------------------------------------------------------------|---------------|
| `api_key`                     | AssemblyAI API key (falls back to `ASSEMBLYAI_API_KEY` env var)           | `None`        |
| `speech_model`                | Model identifier                                                          | `"u3-rt-pro"` |
| `sample_rate`                 | Audio sample rate in Hz                                                   | `16000`       |
| `min_turn_silence`            | Silence (ms) before speculative end-of-turn check                         | API default   |
| `max_turn_silence`            | Maximum silence (ms) before forcing turn end                              | API default   |
| `prompt`                      | Custom transcription prompt (cannot be combined with `keyterms_prompt`)   | `None`        |
| `keyterms_prompt`             | List of terms to boost recognition for (cannot be combined with `prompt`) | `None`        |
| `max_reconnect_attempts`      | Maximum reconnect attempts on transient failures                          | `3`           |
| `reconnect_backoff_initial_s` | Initial backoff delay in seconds                                          | `0.5`         |
| `reconnect_backoff_max_s`     | Maximum backoff delay in seconds                                          | `4.0`         |

## Environment Variables

Set `ASSEMBLYAI_API_KEY` in your environment or pass `api_key` to the constructor.

## Dependencies

- `aiohttp>=3.9.0`
- `vision-agents`

## Docs

- https://www.assemblyai.com/docs/streaming/universal-3-pro
