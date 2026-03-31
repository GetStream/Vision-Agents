# Anam Avatar Plugin for Vision Agents

Add real-time interactive avatar video to your AI agents using [Anam's avatar API](https://anam.ai/).

## Features

- Real-time avatar video and audio synchronized with TTS output
- Audio passthrough mode for high-quality voice reproduction
- Automatic interruption when the user starts speaking
- Works with any TTS provider (ElevenLabs, Cartesia, etc.) or Realtime LLMs
- Configurable video resolution (default 1920x1080)

## Installation

```bash
uv add "vision-agents[anam]"
# or directly
uv add vision-agents-plugins-anam
```

## Quick Start

```python
import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import deepgram, elevenlabs, getstream, gemini
from vision_agents.plugins.anam import AnamAvatarPublisher

load_dotenv()


async def start_avatar_agent():
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Assistant with Avatar", id="agent"),
        instructions="You're a friendly AI assistant.",

        llm=gemini.LLM(),
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),

        processors=[AnamAvatarPublisher()],
    )

    call = await agent.create_call("default", str(uuid4()))

    async with agent.join(call):
        await agent.simple_response("Say hello.")
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_avatar_agent())
```

## Configuration

### Environment Variables

```bash
ANAM_API_KEY=your_anam_api_key
ANAM_AVATAR_ID=your_anam_avatar_id
```

### AvatarPublisher Options

```python
AnamAvatarPublisher(
    avatar_id=None,  # Anam avatar ID (or set ANAM_AVATAR_ID env var)
    api_key=None,  # Anam API key (or set ANAM_API_KEY env var)
    client_options=None,  # Optional Anam ClientOptions for advanced configuration
    connect_timeout=None,  # Seconds to wait for connection (None = wait indefinitely)
    session_ready_timeout=None,  # Seconds to wait for session ready (None = wait indefinitely)
    width=1920,  # Output video width in pixels
    height=1080,  # Output video height in pixels
)
```

## How It Works

1. **Anam Connection**: Connects to Anam's avatar service and waits for the session to become ready
2. **Audio Forwarding**: TTS audio from the agent is resampled to 24kHz mono and streamed to Anam
3. **Avatar Generation**: Anam generates synchronized avatar video and lip-synced audio from the input
4. **Video Streaming**: Avatar video and audio frames are streamed back to call participants via GetStream Edge

## Requirements

- Python 3.10+
- Anam API key and avatar ID (get them at [anam.ai](https://www.anam.ai/))
- GetStream account for video calls
- TTS provider (ElevenLabs, Cartesia, etc.) or Realtime LLM

## License

MIT

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
- [Anam Docs](https://docs.anam.ai/)
