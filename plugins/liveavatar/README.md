# LiveAvatar Plugin for Vision Agents

Real-time interactive avatar via [LiveAvatar](https://docs.liveavatar.com)
(by HeyGen). Uses LITE mode with the custom-agent integration path.

## Features

- Real-time avatar video synchronized with TTS audio
- Works with any TTS provider (Cartesia, ElevenLabs, Deepgram, etc.)
- Supports both standard and Realtime LLMs

## Installation

```bash
uv add "vision-agents[liveavatar]"
# or directly
uv add vision-agents-plugins-liveavatar
```

## Quick Start

```python
import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import Agent, User
from vision_agents.plugins import deepgram, gemini, getstream, liveavatar

load_dotenv()


async def start_avatar_agent():
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Assistant with Avatar", id="agent"),
        instructions="You're a friendly AI assistant.",

        llm=gemini.LLM(),
        tts=deepgram.TTS(),
        stt=deepgram.STT(),

        avatar=liveavatar.Avatar(),
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
LIVEAVATAR_API_KEY=your_liveavatar_api_key
LIVEAVATAR_AVATAR_ID=your_avatar_uuid
```

### Avatar Options

```python
liveavatar.Avatar(
    avatar_id="...",                # LiveAvatar avatar UUID (or set LIVEAVATAR_AVATAR_ID)
    api_key="...",                  # (or set LIVEAVATAR_API_KEY)
    base_url=None,                  # Override https://api.liveavatar.com if needed
    is_sandbox=True,                # Sandbox sessions don't burn credits but are duration-capped. Set False in production.
    max_session_duration=None,      # Seconds; None means LiveAvatar's default
    video_quality="high",           # "low" | "medium" | "high" | "very_high"
    video_encoding="H264",          # "H264" | "VP8"
    width=1920,
    height=1080,
)
```

## Requirements

- Python 3.10+
- LiveAvatar API key — see [docs.liveavatar.com](https://docs.liveavatar.com)
- GetStream account for the user-facing call
- TTS provider or Realtime LLM

## License

MIT

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
- [LiveAvatar Docs](https://docs.liveavatar.com)
- [LITE-mode integration paths](https://docs.liveavatar.com/docs/lite-mode/integration-paths)
