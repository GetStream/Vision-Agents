# Qwen Realtime Plugin for Vision Agents

Qwen3 Realtime LLM integration for Vision Agents framework with native audio output and built-in speech recognition using WebSocket-based realtime communication.

## Features

- **Native audio output**: No TTS service needed - audio comes directly from the model
- **Built-in STT**: Integrated speech-to-text using `gummy-realtime-v1` - no external STT service required
- **Server-side VAD**: Automatic turn detection with configurable silence thresholds
- **Video understanding**: Optional video frame support for multimodal interactions
- **Real-time streaming**: WebSocket-based bidirectional communication for low-latency responses
- **Interruption handling**: Automatic cancellation when user starts speaking

## Installation

```bash
uv add vision-agents[qwen]
```

## Usage

```python
from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, qwen

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Qwen Assistant"),
    instructions="Be helpful and friendly",
    llm=qwen.Realtime(
        model="qwen3-omni-flash-realtime",
        voice="Cherry",
        fps=1,
    ),
    # No STT or TTS needed - Qwen Realtime provides both
)
```

## Configuration

| Parameter | Description | Default | Accepted Values |
|-----------|-------------|---------|----------------|
| `model` | Qwen Realtime model identifier | `"qwen3-omni-flash-realtime"` | Model name string |
| `api_key` | DashScope API key | `None` (from env) | String or `None` |
| `base_url` | WebSocket API base URL | `"wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"` | URL string |
| `voice` | Voice for audio output | `"Cherry"` | Voice name string |
| `fps` | Video frames per second | `1` | Integer |
| `include_video` | Include video frames in requests | `False` | Boolean |
| `video_width` | Video frame width | `1280` | Integer |
| `video_height` | Video frame height | `720` | Integer |

## Environment Variables

Set `DASHSCOPE_API_KEY` in your environment or `.env` file:

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## Example

See `plugins/qwen/example/qwen_realtime_example.py` for a complete working example.

## Dependencies

- vision-agents
- websockets
- aiortc
- av
