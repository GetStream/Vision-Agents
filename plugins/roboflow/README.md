# Roboflow Plugin

Object detection using Roboflow's hosted inference API for vision-agents.

## Installation

```bash
uv add vision-agents-plugins-roboflow
```

## Quick Start

```python
from vision_agents.plugins import roboflow
from vision_agents.core import Agent

# Create processor
processor = roboflow.RoboflowDetectionProcessor(
    api_key="your_api_key",  # or set ROBOFLOW_API_KEY env var
    workspace_id="your-workspace",
    project_id="your-project",
    model_version=1,
    conf_threshold=40,
    fps=5,
)

# Use in an agent
agent = Agent(
    processors=[processor],
    llm=your_llm,
    # ... other components
)
```

## Full Example

See `example/roboflow_example.py` for a complete working example with a video call agent that uses Roboflow detection.

## Configuration

- `api_key`: Your Roboflow API key (or set `ROBOFLOW_API_KEY` env var)
- `workspace_id`: Your Roboflow workspace ID (required)
- `project_id`: Your Roboflow project ID (required)
- `model_version`: Model version number (default: 1)
- `conf_threshold`: Detection confidence threshold 0-100 (default: 40)
- `fps`: Frame processing rate (default: 5)
- `interval`: Processing interval in seconds (default: 0)
- `max_workers`: Thread pool size (default: 10)

## Video Publishing

The processor publishes annotated video frames with bounding boxes drawn on detected objects:

```python
processor = roboflow.RoboflowDetectionProcessor(
    api_key="your_api_key",
    workspace_id="your-workspace",
    project_id="your-project",
)

# The track will show:
# - Green bounding boxes around detected objects
# - Labels with class names and confidence scores
# - Real-time annotation overlay
```

## Testing

```bash
# Run all tests
pytest plugins/roboflow/tests/ -v

# Run specific tests
pytest plugins/roboflow/tests/test_roboflow.py -v
```

## Dependencies

- `vision-agents` - Core framework
- `roboflow>=1.1.0` - Roboflow Python SDK
- `numpy>=2.0.0` - Array operations
- `pillow>=10.4.0` - Image processing
- `opencv-python>=4.8.0` - Video annotation

## Links

- [Roboflow Documentation](https://docs.roboflow.com/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)

