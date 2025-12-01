# Roboflow Plugin Example

This example demonstrates how to use the Roboflow plugin for real-time object detection in a video call.

## Setup

1. **Get API keys:**
    - Roboflow API key: Settings > Roboflow API
    - GetStream API key: https://getstream.io
    - OpenAI API key: https://platform.openai.com/api-keys

2. **Configure environment:**
   ```bash
   cd plugins/roboflow/example
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. **Run the example:**
   ```bash
   uv run python roboflow_example.py
   ```

## What It Does

The agent:

- Joins a video call with object detection enabled
- Processes video frames at 5 FPS using your Roboflow model
- Detects objects based on your trained model
- Annotates the video with bounding boxes and labels
- Can describe what it sees when you ask

## How to test

1. Open the demo UI link that appears in the console
2. Enable your camera
3. Show the agent some football match by sharing the screen with the video.
4. Ask the agent "What do you see?"
5. The video feed will show bounding boxes around detected objects

## Configuration

Edit the example file to customize:

```python
from vision_agents.plugins import roboflow

roboflow_processor = roboflow.RoboflowLocalDetectionProcessor(
    conf_threshold=0.5,  # Confidence threshold (0 - 1.0)
    fps=5,  # Frames per second to process
    annotate=True,  # when True, annotate the detected objects with boxes and labels
    dim_background_factor=0.25  # How much to dim the background around detected objects from 0 to 1.0. 
)
```
