# Moondream Plugin

This plugin provides Moondream 3 vision capabilities for vision-agents, including:
- **Object Detection**: Real-time zero-shot object detection on video streams
- **Visual Question Answering (VQA)**: Answer questions about video frames
- **Image Captioning**: Generate descriptions of video frames

Choose between cloud-hosted or local processing depending on your needs. When running locally, we recommend you do so on CUDA enabled devices.

## Installation

```bash
uv add vision-agents[moondream]
```

## Choosing the Right Component

### Detection Processors

#### CloudDetectionProcessor (Recommended for Most Users)
- **Use when:** You want a simple setup with no infrastructure management
- **Pros:** No model download, no GPU required, automatic updates
- **Cons:** Requires API key, 2 RPS rate limit by default (can be increased)
- **Best for:** Development, testing, low-to-medium volume applications

#### LocalDetectionProcessor (For Advanced Users)
- **Use when:** You need higher throughput, have your own GPU infrastructure, or want to avoid rate limits
- **Pros:** No rate limits, no API costs, full control over hardware
- **Cons:** Requires GPU for best performance, model download on first use, infrastructure management
- **Best for:** Production deployments, high-volume applications, Digital Ocean Gradient AI GPUs, or custom infrastructure

### Vision Language Models (VLM)

#### CloudVLM (Recommended for Most Users)
- **Use when:** You want visual question answering or captioning without managing infrastructure
- **Pros:** No model download, no GPU required, automatic updates
- **Cons:** Requires API key, rate limits apply
- **Best for:** Development, testing, applications requiring VQA or captioning

#### LocalVLM (For Advanced Users)
- **Use when:** You need VQA or captioning with higher throughput or want to avoid rate limits
- **Pros:** No rate limits, no API costs, full control over hardware
- **Cons:** Requires GPU for best performance, model download on first use, infrastructure management
- **Best for:** Production deployments, high-volume applications, or custom infrastructure

## Quick Start

### Using CloudDetectionProcessor (Hosted)

The `CloudDetectionProcessor` uses Moondream's hosted API. By default it has a 2 RPS (requests per second) rate limit and requires an API key. The rate limit can be adjusted by contacting the Moondream team to request a higher limit.

```python
from vision_agents.plugins import moondream
from vision_agents.core import Agent

# Create a cloud processor with detection
processor = moondream.CloudDetectionProcessor(
    api_key="your-api-key",  # or set MOONDREAM_API_KEY env var
    detect_objects="person",  # or ["person", "car", "dog"] for multiple
    fps=30
)

# Use in an agent
agent = Agent(
    processors=[processor],
    llm=your_llm,
    # ... other components
)
```

### Using LocalDetectionProcessor (On-Device)

If you are running on your own infrastructure or using a service like Digital Ocean's Gradient AI GPUs, you can use the `LocalDetectionProcessor` which downloads the model from HuggingFace and runs on device. By default it will use CUDA for best performance. Performance will vary depending on your specific hardware configuration.

**Note:** The moondream3-preview model is gated and requires HuggingFace authentication:
- Request access at https://huggingface.co/moondream/moondream3-preview
- Set `HF_TOKEN` environment variable: `export HF_TOKEN=your_token_here`
- Or run: `huggingface-cli login`

```python
from vision_agents.plugins import moondream
from vision_agents.core import Agent

# Create a local processor (no API key needed)
processor = moondream.LocalDetectionProcessor(
    detect_objects=["person", "car", "dog"],
    conf_threshold=0.3,
    force_cpu=False,  # Auto-detects CUDA, MPS, or CPU
    fps=30
)

# Use in an agent
agent = Agent(
    processors=[processor],
    llm=your_llm,
    # ... other components
)
```

### Detect Multiple Objects

```python
# Detect multiple object types with zero-shot detection
processor = moondream.CloudDetectionProcessor(
    api_key="your-api-key",
    detect_objects=["person", "car", "dog", "basketball"],
    conf_threshold=0.3
)
```

## Vision Language Model (VLM) Quick Start

### Using CloudVLM (Hosted)

The `CloudVLM` uses Moondream's hosted API for visual question answering and captioning. It automatically processes video frames and responds to questions asked via STT (Speech-to-Text).

```python
import asyncio
import os
from dotenv import load_dotenv
from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, elevenlabs, moondream
from vision_agents.core.events import CallSessionParticipantJoinedEvent

load_dotenv()

async def create_agent(**kwargs) -> Agent:
    # Create a cloud VLM for visual question answering
    llm = moondream.CloudVLM(
        api_key=os.getenv("MOONDREAM_API_KEY"),  # or set MOONDREAM_API_KEY env var
        mode="vqa",  # or "caption" for image captioning
    )
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(),
    )
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(2)
            # Ask the agent to describe what it sees
            await agent.simple_response("Describe what you currently see")

    async with agent.join(call):
        await agent.finish()

if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
```

### Using LocalVLM (On-Device)

The `LocalVLM` downloads the model from HuggingFace and runs on device. It supports both VQA and captioning modes.

**Note:** The moondream3-preview model is gated and requires HuggingFace authentication:
- Request access at https://huggingface.co/moondream/moondream3-preview
- Set `HF_TOKEN` environment variable: `export HF_TOKEN=your_token_here`
- Or run: `huggingface-cli login`

```python
from vision_agents.plugins import moondream
from vision_agents.core import Agent

# Create a local VLM (no API key needed)
llm = moondream.LocalVLM(
    mode="vqa",  # or "caption" for image captioning
    force_cpu=False,  # Auto-detects CUDA, MPS, or CPU
)

# Use in an agent
agent = Agent(
    llm=llm,
    tts=your_tts,
    stt=your_stt,
    # ... other components
)
```

### VLM Modes

The VLM supports two modes:

- **`"vqa"`** (Visual Question Answering): Answers questions about video frames. Questions come from STT transcripts.
- **`"caption"`** (Image Captioning): Generates descriptions of video frames automatically.

```python
# VQA mode - answers questions about frames
llm = moondream.CloudVLM(
    api_key="your-api-key",
    mode="vqa"
)

# Caption mode - generates automatic descriptions
llm = moondream.CloudVLM(
    api_key="your-api-key",
    mode="caption"
)
```

## Configuration

### CloudDetectionProcessor Parameters

- `api_key`: str - API key for Moondream Cloud API. If not provided, will attempt to read from `MOONDREAM_API_KEY` environment variable.
- `detect_objects`: str | List[str] - Object(s) to detect using zero-shot detection. Can be any object name like "person", "car", "basketball". Default: `"person"`
- `conf_threshold`: float - Confidence threshold for detections (default: 0.3)
- `fps`: int - Frame processing rate (default: 30)
- `interval`: int - Processing interval in seconds (default: 0)
- `max_workers`: int - Thread pool size for CPU-intensive operations (default: 10)

**Rate Limits:** By default, the Moondream Cloud API has a 2rps (requests per second) rate limit. Contact the Moondream team to request a higher limit.

### LocalDetectionProcessor Parameters

- `detect_objects`: str | List[str] - Object(s) to detect using zero-shot detection. Can be any object name like "person", "car", "basketball". Default: `"person"`
- `conf_threshold`: float - Confidence threshold for detections (default: 0.3)
- `fps`: int - Frame processing rate (default: 30)
- `interval`: int - Processing interval in seconds (default: 0)
- `max_workers`: int - Thread pool size for CPU-intensive operations (default: 10)
- `force_cpu`: bool - If True, force CPU usage even if CUDA/MPS is available. Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU. We recommend running on CUDA for best performance. (default: False)
- `model_name`: str - Hugging Face model identifier (default: "moondream/moondream3-preview")
- `options`: AgentOptions - Model directory configuration. If not provided, uses default which defaults to tempfile.gettempdir()

**Performance:** Performance will vary depending on your hardware configuration. CUDA is recommended for best performance on NVIDIA GPUs. The model will be downloaded from HuggingFace on first use.

### CloudVLM Parameters

- `api_key`: str - API key for Moondream Cloud API. If not provided, will attempt to read from `MOONDREAM_API_KEY` environment variable.
- `mode`: Literal["vqa", "caption"] - "vqa" for visual question answering or "caption" for image captioning (default: "vqa")
- `max_workers`: int - Thread pool size for CPU-intensive operations (default: 10)

**Rate Limits:** By default, the Moondream Cloud API has rate limits. Contact the Moondream team to request higher limits.

### LocalVLM Parameters

- `mode`: Literal["vqa", "caption"] - "vqa" for visual question answering or "caption" for image captioning (default: "vqa")
- `max_workers`: int - Thread pool size for async operations (default: 10)
- `force_cpu`: bool - If True, force CPU usage even if CUDA/MPS is available. Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU. Note: MPS is automatically converted to CPU due to model compatibility. We recommend running on CUDA for best performance. (default: False)
- `model_name`: str - Hugging Face model identifier (default: "moondream/moondream3-preview")
- `options`: AgentOptions - Model directory configuration. If not provided, uses default_agent_options()

**Performance:** Performance will vary depending on your hardware configuration. CUDA is recommended for best performance on NVIDIA GPUs. The model will be downloaded from HuggingFace on first use.

## Video Publishing

The processor publishes annotated video frames with bounding boxes drawn on detected objects:

```python
processor = moondream.CloudDetectionProcessor(
    api_key="your-api-key",
    detect_objects=["person", "car"]
)

# The track will show:
# - Green bounding boxes around detected objects
# - Labels with confidence scores
# - Real-time annotation overlay
```

## Testing

The plugin includes comprehensive tests:

```bash
# Run all tests
pytest plugins/moondream/tests/ -v

# Run specific test categories
pytest plugins/moondream/tests/ -k "inference" -v
pytest plugins/moondream/tests/ -k "annotation" -v
```

## Dependencies

### Required
- `vision-agents` - Core framework
- `moondream` - Moondream SDK for cloud API (CloudDetectionProcessor and CloudVLM)
- `numpy>=2.0.0` - Array operations
- `pillow>=10.0.0` - Image processing
- `opencv-python>=4.8.0` - Video annotation
- `aiortc` - WebRTC support

### Local Components Additional Dependencies
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace transformers library for model loading

**Note:** LocalDetectionProcessor and LocalVLM both require these dependencies. We recommend only running the model locally on CUDA devices. 

## Links

- [Moondream Documentation](https://docs.moondream.ai/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)


