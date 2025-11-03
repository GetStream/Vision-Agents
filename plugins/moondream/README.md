# Moondream Plugin

This plugin provides Moondream 3 vision capabilities for vision-agents, enabling real-time zero-shot object detection on video streams using the Moondream Cloud API.

## Features

- **Object Detection** with bounding boxes drawn on video frames
- **Zero-shot Detection** - Detect any object by name using natural language
- **LLM Integration** - Expose visual understanding to conversation agents via `state()` method
- **High Performance** - Async processing with ThreadPoolExecutor for CPU-intensive operations
- **Frame Queuing** - Latest-N queue pattern for smooth video publishing
- **Moondream Cloud API** - Powered by state-of-the-art vision model

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
    api_key="your-api-key",
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

### Detect Multiple Objects

```python
# Detect multiple object types with zero-shot detection
processor = moondream.CloudDetectionProcessor(
    api_key="your-api-key",
    detect_objects=["person", "car", "dog", "basketball"],
    conf_threshold=0.3
)

# Access results for LLM
state = processor.state()
print(state["detections_summary"])  # "Detected: 2 persons, 1 car"
print(state["detections_count"])  # Total number of detections
print(state["last_image"])  # PIL Image for vision models
```

## Configuration

### Parameters

- `api_key`: str - API key for Moondream Cloud API. If not provided, will attempt to read from `MOONDREAM_API_KEY` environment variable.
- `detect_objects`: str | List[str] - Object(s) to detect using zero-shot detection. Can be any object name like "person", "car", "basketball". Default: `"person"`
- `conf_threshold`: float - Confidence threshold for detections (default: 0.3)
- `fps`: int - Frame processing rate (default: 30)
- `interval`: int - Processing interval in seconds (default: 0)
- `max_workers`: int - Thread pool size for CPU-intensive operations (default: 4)

## LLM Integration

The processor exposes visual understanding to LLMs via the `state()` method:

```python
processor = moondream.CloudDetectionProcessor(
    api_key="your-api-key",
    detect_objects=["person", "car", "dog"]
)

# After processing frames...
state = processor.state()

# Available state fields:
# - last_frame_timestamp: float - When the frame was processed
# - last_image: PIL.Image - For vision-capable LLMs
# - detections_summary: str - Human-readable summary (e.g., "Detected: 2 persons, 1 car")
# - detections_count: int - Total number of objects detected

# Use in conversation
print(f"Objects detected: {state['detections_summary']}")
print(f"Total objects: {state['detections_count']}")
```

## Video Publishing

The processor publishes annotated video frames with bounding boxes drawn on detected objects:

```python
processor = moondream.CloudDetectionProcessor(
    api_key="your-api-key",
    detect_objects=["person", "car"]
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
- `moondream` - Moondream SDK for cloud API
- `numpy>=2.0.0` - Array operations
- `pillow>=10.0.0` - Image processing
- `opencv-python>=4.8.0` - Video annotation
- `aiortc` - WebRTC support

## Links

- [Moondream Documentation](https://docs.moondream.ai/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)


