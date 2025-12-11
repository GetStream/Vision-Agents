# Qwen Omni Example

This example demonstrates how to use Qwen Omni with Vision Agents for conversations with native audio output.

Qwen Omni is a multimodal model that generates native audio output (no TTS needed), but requires STT for speech input since Qwen's API only accepts audio via URLs, not inline data.

## Features

- **Native audio output**: Qwen generates audio directly (no TTS service needed)
- **Streaming responses**: Real-time text and audio streaming
- **STT required**: Use any STT service (e.g., Deepgram) for speech-to-text input

## Installation

```bash
uv add vision-agents[qwen]
```

## Quick Start

1. Copy the environment file and add your API keys:

```bash
cp .env.example .env
```

2. Set your API keys in `.env`:

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

3. Run the example:

```bash
uv run python qwen_omni_example.py
```

## Code Example

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, qwen, deepgram

async def create_agent(**kwargs) -> Agent:
    # Initialize Qwen Omni LLM
    llm = qwen.QwenOmni(
        model="qwen3-omni-flash",
        voice="Cherry",
    )

    # Create an agent with STT for speech input
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Qwen Assistant", id="agent"),
        instructions="You are a helpful AI assistant.",
        llm=llm,
        stt=deepgram.STT(),  # Required for speech input
        # No TTS needed - Qwen provides native audio output
    )
    return agent
```

## Configuration

### Environment Variables

- **`ALIBABA_API_KEY`**: Your DashScope/Alibaba API key (required)
- **`DEEPGRAM_API_KEY`**: Your Deepgram API key (required for STT)
- **`STREAM_API_KEY`**: Your Stream API key (required for video calls)
- **`STREAM_API_SECRET`**: Your Stream API secret (required for video calls)

### QwenOmni Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Qwen Omni model identifier | `"qwen3-omni-flash"` |
| `api_key` | DashScope API key | `None` (from env) |
| `base_url` | API base URL | DashScope endpoint |
| `voice` | Voice for audio output | `"Cherry"` |
| `audio_format` | Audio format | `"wav"` |
| `fps` | Video frames per second | `1` |
| `include_video` | Include video frames | `False` |

## How It Works

1. **Speech Input**: Users speak, and STT (e.g., Deepgram) transcribes speech to text.

2. **Text Processing**: The transcribed text is sent to Qwen Omni via the Chat Completions API.

3. **API Request**: The plugin uses `modalities=["text", "audio"]` to request both text and audio responses.

4. **Audio Output**: The streaming response includes base64-encoded audio data which is decoded and played back through the video call (no TTS needed).

## Requirements

- Python 3.10+
- DashScope API key
- Stream API credentials
- `vision-agents` framework

## Notes

- Qwen Omni uses the Chat Completions API format but returns audio output
- STT is required for speech input (Qwen's API requires URLs for audio, not inline data)
- Native audio output eliminates the need for a separate TTS service
- Video frames are supported via base64 encoding (set `include_video=True` to enable)
