# Qwen Omni Plugin for Vision Agents

Qwen Omni LLM integration for Vision Agents framework with native audio output using the Chat Completions API.

## Features

- **Native audio output**: No TTS service needed - audio comes directly from the model
- **STT compatible**: Works with any STT service (Deepgram, etc.) for speech input
- **Video understanding**: Optional video frame support (as base64 images)
- Streaming text and audio responses
- Compatible with OpenAI Chat Completions API format

## Installation

```bash
uv add vision-agents[qwen]
```

## Usage

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, qwen, deepgram

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Qwen Assistant"),
    instructions="Be helpful and friendly",
    llm=qwen.QwenOmni(
        model="qwen3-omni-flash",
        voice="Cherry",
    ),
    stt=deepgram.STT(),  # STT required for speech input
    # No TTS needed - Qwen provides native audio output
)
```

## Configuration

| Parameter | Description | Default | Accepted Values |
|-----------|-------------|---------|----------------|
| `model` | Qwen Omni model identifier | `"qwen3-omni-flash"` | Model name string |
| `api_key` | DashScope API key | `None` (from env) | String or `None` |
| `base_url` | API base URL | `"https://dashscope-intl.aliyuncs.com/compatible-mode/v1"` | URL string |
| `voice` | Voice for audio output | `"Cherry"` | Voice name string |
| `audio_format` | Audio format for output | `"wav"` | `"wav"` |
| `fps` | Video frames per second | `1` | Integer |
| `include_video` | Include video frames in requests | `False` | Boolean |
| `client` | Custom AsyncOpenAI client | `None` | `AsyncOpenAI` or `None` |

## Dependencies

- vision-agents
- openai>=2.7.2
- numpy
- soundfile
