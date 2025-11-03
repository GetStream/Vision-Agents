# Moondream Plugin

This plugin provides Moondream 3 detection capabilities for vision-agents, enabling real-time zero-shot object detection on video streams. Choose between cloud-hosted or local processing depending on your needs.

## Installation

```bash
uv add vision-agents-plugins-moondream
```

## Choosing the Right Processor

### CloudDetectionProcessor (Recommended for Most Users)
- **Use when:** You want a simple setup with no infrastructure management
- **Pros:** No model download, no GPU required, automatic updates
- **Cons:** Requires API key, 2 RPS rate limit by default (can be increased)
- **Best for:** Development, testing, low-to-medium volume applications

### LocalDetectionProcessor (For Advanced Users)
- **Use when:** You need higher throughput, have your own GPU infrastructure, or want to avoid rate limits
- **Pros:** No rate limits, no API costs, full control over hardware
- **Cons:** Requires GPU for best performance, model download on first use, infrastructure management
- **Best for:** Production deployments, high-volume applications, Digital Ocean Gradient AI GPUs, or custom infrastructure

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
    device="cuda",  # Auto-detects CUDA, MPS, or CPU
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
- `device`: str - Device to run inference on ('cuda', 'mps', or 'cpu'). Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU. Default: `None` (auto-detect)
- `model_name`: str - Hugging Face model identifier (default: "moondream/moondream3-preview")
- `options`: AgentOptions - Model directory configuration. If not provided, uses default which defaults to tempfile.gettempdir()

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
pytest plugins/moondream/tests/ -k "state" -v
```

## Dependencies

### Required
- `vision-agents` - Core framework
- `moondream` - Moondream SDK for cloud API (CloudDetectionProcessor only)
- `numpy>=2.0.0` - Array operations
- `pillow>=10.0.0` - Image processing
- `opencv-python>=4.8.0` - Video annotation
- `aiortc` - WebRTC support

### LocalDetectionProcessor Additional Dependencies
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace transformers library for model loading

## Links

- [Moondream Documentation](https://docs.moondream.ai/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)


