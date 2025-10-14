# Moondream Plugin

This plugin provides Moondream 3 vision capabilities for vision-agents, enabling real-time object detection, visual question answering, counting, and image captioning with video stream processing.

## Features

- **Object Detection** with bounding boxes drawn on video frames
- **Visual Question Answering (VQA)** - Ask questions about what's in the frame
- **Counting** - Count specific objects in images
- **Image Captioning** - Generate natural language descriptions
- **Multiple Inference Modes**: Cloud API, local inference, and FAL.ai
- **LLM Integration** - Expose visual understanding to conversation agents via `state()` method
- **High Performance** - Async processing with ThreadPoolExecutor for CPU-intensive operations
- **Frame Queuing** - Latest-N queue pattern for smooth video publishing

## Installation

```bash
uv add vision-agents-plugins-moondream
```

## Quick Start

### Basic Object Detection

```python
from vision_agents.plugins import moondream

# Create a Moondream processor with detection
processor = moondream.MoondreamProcessor(
    mode="cloud",
    skills=["detection"],
    api_key="your-api-key",
    fps=30
)

# Use in an agent
agent = Agent(
    processors=[processor],
    llm=your_llm,
    # ... other components
)
```

### Visual Question Answering

```python
processor = moondream.MoondreamProcessor(
    mode="cloud",
    skills=["vqa"],
    vqa_prompt="What objects are in this scene?",
    api_key="your-api-key"
)
```

### Multiple Skills

```python
processor = moondream.MoondreamProcessor(
    mode="cloud",
    skills=["detection", "vqa", "caption", "counting"],
    vqa_prompt="What's happening in this scene?",
    api_key="your-api-key"
)

# Access results for LLM
state = processor.state()
print(state["vqa_response"])  # Answer to the question
print(state["caption"])  # Image description
print(state["detections_summary"])  # "Detected: 2 persons, 1 car"
print(state["count"])  # Number of objects
```

## Configuration

### Required Parameters

- `mode`: Inference mode
  - `"cloud"` - Moondream Cloud API (recommended for production)
  - `"local"` - Local model inference (requires model files)
  - `"fal"` - FAL.ai API

### Optional Parameters

- `skills`: List[str] - Skills to enable (default: `["detection"]`)
  - `"detection"` - Object detection with bounding boxes
  - `"vqa"` - Visual question answering
  - `"counting"` - Count objects
  - `"caption"` - Generate image captions

- `api_key`: str - API key for cloud/FAL modes
- `model_path`: str - Path to local model (for local mode only)
- `vqa_prompt`: str - Default question for VQA (default: "What do you see?")
- `conf_threshold`: float - Confidence threshold for detections (default: 0.3)
- `fps`: int - Frame processing rate (default: 30)
- `interval`: int - Processing interval in seconds (default: 0)
- `max_workers`: int - Thread pool size (default: 4)
- `device`: str - Device for local inference: "cpu" or "cuda" (default: "cpu")

## Inference Modes

### Cloud API Mode

Best for production with managed infrastructure:

```python
processor = moondream.MoondreamProcessor(
    mode="cloud",
    api_key="your-moondream-api-key",
    skills=["detection", "vqa"]
)
```

### Local Mode

Run inference on your own hardware:

```python
processor = moondream.MoondreamProcessor(
    mode="local",
    model_path="/path/to/moondream/model",
    device="cuda",  # or "cpu"
    skills=["detection"]
)
```

**Note:** Local mode requires downloading Moondream 3 model files and installing the moondream SDK.

### FAL.ai Mode

Use FAL.ai's managed inference:

```python
processor = moondream.MoondreamProcessor(
    mode="fal",
    api_key="your-fal-api-key",
    skills=["detection", "caption"]
)
```

## LLM Integration

The processor exposes visual understanding to LLMs via the `state()` method:

```python
processor = moondream.MoondreamProcessor(
    mode="cloud",
    skills=["detection", "vqa", "caption"],
    vqa_prompt="What objects do you see?"
)

# After processing frames...
state = processor.state()

# Available state fields:
# - last_frame_timestamp: float
# - last_image: PIL.Image (for vision-capable LLMs)
# - vqa_response: str (answer to VQA prompt)
# - caption: str (image description)
# - count: int (object count)
# - detections_summary: str (e.g., "Detected: 2 persons, 1 car")
# - detections_count: int (number of detections)

# Use in conversation
print(f"I see: {state['caption']}")
print(f"Objects detected: {state['detections_summary']}")
```

## Video Publishing

The processor publishes annotated video frames with bounding boxes drawn on detected objects:

```python
processor = moondream.MoondreamProcessor(
    mode="cloud",
    skills=["detection"]
)

# Get video track for publishing
video_track = processor.publish_video_track()

# The track will show:
# - Green bounding boxes around detected objects
# - Labels with confidence scores
# - Real-time annotation overlay
```

## Moondream 3 Performance

Moondream 3 is a mixture-of-experts vision language model with state-of-the-art performance:

### Model Stats
- **9B total params, 2B active params** - Fast inference with large capacity
- **32k context window** - Process large images and long sequences
- **Native grounded reasoning** - Precise object localization

### Benchmark Results

| Task | Moondream 3 | GPT 5 | Gemini 2.5-Flash | Claude 4 Sonnet |
|------|-------------|-------|------------------|-----------------|
| **Object Detection** | | | | |
| refcocog | **88.6** | 49.8 | 75.1 | 26.2 |
| refcoco+ | **81.8** | 46.3 | 70.2 | 23.4 |
| refcoco | **91.1** | 57.2 | 75.8 | 30.1 |
| **Counting** | | | | |
| CountbenchQA | **93.2** | 89.3 | 81.2 | 90.1 |
| **Document Understanding** | | | | |
| ChartQA | **86.6** | 85* | 79.5 | 74.3* |
| DocVQA | 88.3 | 89* | **94.2** | 89.5* |

## Testing

The plugin includes comprehensive tests:

```bash
# Run all tests
pytest plugins/moondream/tests/ -v

# Run specific test categories
pytest plugins/moondream/tests/ -k "inference" -v
pytest plugins/moondream/tests/ -k "annotation" -v
pytest plugins/moondream/tests/ -k "state" -v
```

## Dependencies

### Required
- `vision-agents` - Core framework
- `numpy>=2.0.0` - Array operations
- `pillow>=10.0.0` - Image processing
- `opencv-python>=4.8.0` - Video annotation
- `aiohttp>=3.9.0` - Async HTTP for API modes

### Optional
- `moondream>=0.3.0` - For local inference mode

## Links

- [Moondream Documentation](https://docs.moondream.ai/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)


