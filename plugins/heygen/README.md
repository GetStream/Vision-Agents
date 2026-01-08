# HeyGen Avatar Plugin for Vision Agents

Add realistic avatar video to your AI agents using HeyGen's streaming avatar API.

## Features

- 🎭 **Realistic Avatars**: Use HeyGen's high-quality avatars with natural movements
- 🎤 **Automatic Lip-Sync**: Avatar automatically syncs with audio from any TTS provider
- 🚀 **WebRTC Streaming**: Low-latency real-time video streaming via WebRTC
- 🔌 **Easy Integration**: Works seamlessly with Vision Agents framework
- 🎨 **Customizable**: Configure avatar, quality, resolution, and more

## Installation

```bash
pip install vision-agents-plugins-heygen
```

Or with uv:

```bash
uv pip install vision-agents-plugins-heygen
```

## Quick Start

```python
import asyncio
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import cartesia, deepgram, getstream, gemini, heygen
from vision_agents.plugins.heygen import VideoQuality

load_dotenv()

async def start_avatar_agent():
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="AI Assistant with Avatar", id="agent"),
        instructions="You're a friendly AI assistant.",
        
        llm=gemini.LLM("gemini-2.0-flash"),
        tts=cartesia.TTS(),
        stt=deepgram.STT(),
        
        # Add HeyGen avatar
        processors=[
            heygen.AvatarPublisher(
                avatar_id="default",
                quality=VideoQuality.HIGH
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

Set your HeyGen API key:

```bash
HEYGEN_API_KEY=your_heygen_api_key_here
```

### AvatarPublisher Options

```python
from vision_agents.plugins.heygen import VideoQuality

heygen.AvatarPublisher(
    avatar_id="default",           # HeyGen avatar ID
    quality=VideoQuality.HIGH,    # Video quality: VideoQuality.LOW, VideoQuality.MEDIUM, or VideoQuality.HIGH
    resolution=(1920, 1080),       # Output resolution (width, height)
    api_key=None,                  # Optional: override env var
)
```

## Usage Examples

### With Realtime LLM

```python
from vision_agents.plugins import gemini, heygen, getstream

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Realtime Avatar AI"),
    instructions="Be conversational and responsive.",
    
    llm=gemini.Realtime(fps=2),  # No separate TTS needed
    
    processors=[
        heygen.AvatarPublisher(avatar_id="professional_presenter")
    ]
)

call = agent.edge.client.video.call("default", str(uuid4()))

async with agent.join(call):
    await agent.finish()
```

### With Multiple Processors

```python
from vision_agents.plugins import ultralytics, heygen

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Fitness Coach"),
    instructions="Analyze user poses and provide feedback.",
    
    llm=gemini.Realtime(fps=3),
    
    processors=[
        # Process incoming user video
        ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt"),
        # Publish avatar video
        heygen.AvatarPublisher(avatar_id="fitness_trainer")
    ]
)
```

## How It Works

1. **Connection**: Establishes WebRTC connection to HeyGen's streaming API
2. **Audio Input**: Receives audio from your TTS provider or Realtime LLM
3. **Avatar Generation**: HeyGen generates avatar video with lip-sync
4. **Video Streaming**: Streams avatar video to call participants via GetStream Edge

## Requirements

- Python 3.10+
- HeyGen API key (get one at [heygen.com](https://heygen.com))
- GetStream account for video calls
- TTS provider (Cartesia, ElevenLabs, etc.) or Realtime LLM

## Troubleshooting

### Connection Issues

If you experience connection problems:

1. Check your HeyGen API key is valid
2. Ensure you have network access to HeyGen's servers
3. Check firewall settings for WebRTC traffic

### Video Quality

To optimize video quality:

- Use `quality=VideoQuality.HIGH` for best results
- Increase resolution if bandwidth allows
- Ensure stable internet connection

## API Reference

### AvatarPublisher

Main class for publishing HeyGen avatar video.

**Methods:**
- `publish_video_track()`: Returns video track for streaming
- `state()`: Returns current state information
- `close()`: Clean up resources

## License

MIT

## Links

- [Documentation](https://visionagents.ai/)
- [GitHub](https://github.com/GetStream/Vision-Agents)
- [HeyGen API Docs](https://docs.heygen.com/docs/streaming-api)

