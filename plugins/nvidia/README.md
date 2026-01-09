# NVIDIA Plugin for Vision Agents

NVIDIA VLM integration for Vision Agents. Supports vision language models through NVIDIA's Chat Completions API with NVCF asset management.

## Features

- Video understanding: Automatically buffers and forwards video frames to NVIDIA VLM models
- Streaming responses: Supports streaming text responses with real-time chunk events
- Asset management: Automatically uploads frames as assets and cleans them up after use
- Configurable frame rate and buffer duration for optimal performance

## Installation

```bash
uv add vision-agents[nvidia]
```

## Configuration

Set your NVIDIA API key:

```bash
export NVIDIA_API_KEY=your_nvidia_api_key
```

## Usage

### Basic VLM Usage

```python
from vision_agents.plugins import nvidia

vlm = nvidia.VLM(
    model="nvidia/cosmos-reason2-8b",
    fps=1,
    frame_buffer_seconds=10,
)

# VLM automatically buffers video frames when used with an Agent
response = await vlm.simple_response("What do you see?")
print(response.text)
```

### With Custom Configuration

```python
from vision_agents.plugins import nvidia

vlm = nvidia.VLM(
    model="nvidia/cosmos-reason2-8b",
    api_key="your-api-key",
    fps=2,
    frame_buffer_seconds=15,
    frame_width=1280,
    frame_height=720,
    max_tokens=2048,
    temperature=0.3,
    top_p=0.8,
)
```

## Configuration

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `model` | NVIDIA model ID | `"nvidia/cosmos-reason2-8b"` | str |
| `api_key` | NVIDIA API token (or use `NVIDIA_API_KEY` env var) | `None` | Optional[str] |
| `fps` | Frames per second to buffer | `1` | int |
| `frame_buffer_seconds` | Number of seconds to buffer | `10` | int |
| `frame_width` | Width of video frames to send | `800` | int |
| `frame_height` | Height of video frames to send | `600` | int |
| `max_tokens` | Maximum response tokens | `1024` | int |
| `temperature` | Temperature for sampling | `0.2` | float |
| `top_p` | Top-p sampling parameter | `0.7` | float |
| `frames_per_second` | Frames per second for video models | `8` | int |

## Dependencies

- `vision-agents`: Core Vision Agents framework
- `aiohttp>=3.9.0`: Async HTTP client for API requests
