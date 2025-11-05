# Roboflow Plugin Example

This example demonstrates how to use the Roboflow plugin for real-time object detection in a video call.

## Setup

1. **Create a Roboflow account and project:**
   - Go to https://app.roboflow.com
   - Create or upload a trained model
   - Note your workspace ID, project ID, and model version

2. **Get API keys:**
   - Roboflow API key: Settings > Roboflow API
   - GetStream API key: https://getstream.io
   - Gemini API key: https://ai.google.dev
   - Cartesia API key: https://cartesia.ai
   - Deepgram API key: https://deepgram.com

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Run the example:**
   ```bash
   cd /Users/mkahan/Development/Vision-Agents
   uv run python plugins/roboflow/example/roboflow_example.py
   ```

## What It Does

The agent:
- Joins a video call with object detection enabled
- Processes video frames at 5 FPS using your Roboflow model
- Detects objects based on your trained model
- Annotates the video with bounding boxes and labels
- Can describe what it sees when you ask

## Configuration

Edit the example file to customize:

```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    workspace_id="your-workspace",
    project_id="your-project",
    model_version=1,
    conf_threshold=40,  # Confidence threshold (0-100)
    fps=5,              # Frames per second to process
)
```

## Testing Locally

1. Open the demo UI link that appears in the console
2. Enable your camera
3. Show objects to the camera
4. Ask the agent "What do you see?"
5. The video feed will show bounding boxes around detected objects

## Rate Limits

Roboflow's hosted API has rate limits. The example uses 5 FPS by default to be API-friendly. 

For higher throughput, consider:
- Requesting a higher rate limit from Roboflow
- Using a dedicated Roboflow inference server
- Reducing the FPS further if you hit limits

