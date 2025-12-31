# Roboflow Object Detection Example

This example demonstrates how to use the Roboflow plugin for real-time object detection with Vision Agents.

## Setup

1. Install dependencies:

```bash
cd plugins/roboflow/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
cp env.example .env
# Edit .env with your actual credentials
```

## Running the Example

```bash
uv run roboflow_example.py
```

The agent will:
1. Connect to GetStream
2. Join a video call with object detection enabled
3. Process video frames at 5 FPS using RF-DETR
4. Annotate the video with bounding boxes around detected objects

## Customization

### Detection Classes

Specify which objects to detect:

```python
processor = roboflow.RoboflowLocalDetectionProcessor(
    classes=["person", "car", "dog"],  # Only these classes
    conf_threshold=0.5,
    fps=5,
)
```

### Using Cloud Inference

For cloud-based detection with Roboflow Universe models:

```python
processor = roboflow.RoboflowCloudDetectionProcessor(
    api_key="your_api_key",  # or set ROBOFLOW_API_KEY env var
    model_id="your-model-id/version",
    conf_threshold=0.5,
    fps=3,
)
```

### Event Handling

React to detection events:

```python
@agent.events.subscribe
async def on_detection(event: roboflow.DetectionCompletedEvent):
    for obj in event.objects:
        print(f"Detected {obj['label']} with confidence {obj['confidence']}")
```

## Additional Resources

- [Roboflow Documentation](https://docs.roboflow.com/)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Vision Agents Documentation](https://visionagents.ai/)
