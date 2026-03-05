# LemonSlice Avatar Plugin for Vision Agents

Add real-time interactive avatar video to your AI agents using LemonSlice's self-managed API.

## Features

- Real-time avatar video synchronized with TTS audio
- Works with any TTS provider (Cartesia, ElevenLabs, etc.)
- Supports both standard and Realtime LLMs
- Customizable avatar expressions via agent prompts

## Installation

```bash
pip install vision-agents[lemonslice]
```

Or with uv:

```bash
uv pip install vision-agents[lemonslice]
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

        processors=[
            lemonslice.LemonSliceAvatarPublisher(
                agent_id="your-avatar-id",
            )
        ]
    )

    call = agent.edge.client.video.call("default", str(uuid4()))

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
# LemonSlice uses Livekit as a transport for audio and video
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

### AvatarPublisher Options

```python
lemonslice.LemonSliceAvatarPublisher(
    agent_id="your-avatar-id",  # LemonSlice agent ID
    agent_image_url=None,  # Or provide a custom image URL (368x560px)
    agent_prompt=None,  # Prompt to influence avatar expressions/movements
    api_key=None,  # Optional: override LEMONSLICE_API_KEY env var
    idle_timeout=None,  # Session timeout in seconds
    livekit_url=None,  # Optional: override LIVEKIT_URL env var
    livekit_api_key=None,  # Optional: override LIVEKIT_API_KEY env var
    livekit_api_secret=None,  # Optional: override LIVEKIT_API_SECRET env var
    width=1920,  # Output video width in pixels
    height=1080,  # Output video height in pixels
)
```

## How It Works

1. **LemonSlice Session**: Creates a session via LemonSlice API, and joins the LiveKit room as a participant
2. **Audio Forwarding**: TTS audio is captured and sent to LemonSlice via the room
3. **Avatar Generation**: LemonSlice generates synchronized avatar video and audio
4. **Video Streaming**: Avatar video is streamed to call participants via GetStream Edge

## Requirements

- Python 3.10+
- LemonSlice API key (get one at [lemonslice.com](https://lemonslice.com))
- LiveKit server (cloud or self-hosted)
- GetStream account for video calls
- TTS provider (Cartesia, ElevenLabs, etc.) or Realtime LLM

## License

MIT

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
- [LemonSlice Docs](https://lemonslice.com/docs/self-managed/overview)
