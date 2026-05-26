# LemonSlice Avatar Plugin for Vision Agents

Add real-time interactive avatar video to your AI agents using LemonSlice's self-managed API.

## Features

- Real-time avatar video synchronized with TTS audio
- Works with any TTS provider (Cartesia, ElevenLabs, etc.)
- Supports both standard and Realtime LLMs
- Customizable avatar expressions via agent prompts

## Installation

```bash
uv add "vision-agents[lemonslice]"
# or directly
uv add vision-agents-plugins-lemonslice
```

## Quick Start

```python
import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import cartesia, deepgram, getstream, gemini, lemonslice

load_dotenv()


async def start_avatar_agent():
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Assistant with Avatar", id="agent"),
        instructions="You're a friendly AI assistant.",

        llm=gemini.LLM(),
        tts=cartesia.TTS(),
        stt=deepgram.STT(),

        avatar=lemonslice.Avatar(agent_id="your-avatar-id"),
    )

    call = await agent.create_call("default", str(uuid4()))

    async with agent.join(call):
        await agent.simple_response("Hello! I'm your AI assistant with an avatar.")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_avatar_agent())
```

## Configuration

### Environment Variables

```bash
LEMONSLICE_API_KEY=your_lemonslice_api_key
LEMONSLICE_AGENT_ID=your_agent_id
# Or, instead of LEMONSLICE_AGENT_ID:
# LEMONSLICE_AGENT_IMAGE_URL=https://example.com/avatar.png

# LemonSlice uses Stream as the transport for audio and video
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

### Avatar Options

```python
lemonslice.Avatar(
    agent_id="your-avatar-id",  # LemonSlice agent ID (or set LEMONSLICE_AGENT_ID env var)
    agent_image_url=None,  # Custom image URL, 368x560px (or set LEMONSLICE_AGENT_IMAGE_URL env var)
    agent_prompt=None,  # Prompt to influence avatar expressions/movements
    api_key=None,  # Optional: override LEMONSLICE_API_KEY env var
    idle_timeout=None,  # Session timeout in seconds
    stream_api_key=None,  # Optional: override STREAM_API_KEY env var
    stream_api_secret=None,  # Optional: override STREAM_API_SECRET env var
    width=1280,  # Output video width in pixels
    height=720,  # Output video height in pixels
    fps=30,  # Output video frame rate
    buffer_seconds=1.0,  # Max video buffer depth in seconds
)
```

## How It Works

1. **LemonSlice Session**: Creates a session via LemonSlice API, and joins the Stream call as a participant
2. **Audio Forwarding**: TTS audio is captured and sent to LemonSlice via the Stream call
3. **Avatar Generation**: LemonSlice generates synchronized avatar video and audio
4. **Video Streaming**: Avatar video is streamed to call participants via GetStream Edge

## Requirements

- Python 3.10+
- LemonSlice API key (get one at [lemonslice.com](https://lemonslice.com))
- GetStream account for video calls
- TTS provider (Cartesia, ElevenLabs, etc.) or Realtime LLM

## License

MIT

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
- [LemonSlice Docs](https://lemonslice.com/docs/self-managed/overview)
