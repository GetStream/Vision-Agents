# Roboflow Object Detection Example

This example demonstrates real-time object detection in video calls using the Roboflow plugin.

## What It Does

- Joins a Stream video call as an AI agent
- Processes video frames at 5 FPS using your trained Roboflow model
- Detects objects based on your model's training
- Annotates video with bounding boxes and labels
- Describes what it sees when you ask

## Prerequisites

1. **Roboflow Account & Model**
   - Sign up at https://app.roboflow.com
   - Train a model or use an existing one
   - Note your workspace ID, project ID, and model version

2. **API Keys**
   - Roboflow API key (Settings > Roboflow API)
   - GetStream API key (https://getstream.io)
   - Gemini API key (https://ai.google.dev)
   - Cartesia API key (https://cartesia.ai)
   - Deepgram API key (https://deepgram.com)

## Setup

1. **Install dependencies:**
   ```bash
   cd examples/other_examples/plugins_examples/roboflow_detection
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys and Roboflow project details
   ```

3. **Run the example:**
   ```bash
   uv run python main.py
   ```

## Configuration

### Roboflow Settings

Edit `main.py` to customize detection:

```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    workspace_id="your-workspace",
    project_id="your-project",
    model_version=1,
    conf_threshold=40,  # Confidence threshold (0-100)
    fps=5,              # Frames per second to process
)
```

### Environment Variables

Required variables in `.env`:

```bash
# Roboflow
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE=your-workspace
ROBOFLOW_PROJECT=your-project
ROBOFLOW_VERSION=1

# GetStream
STREAM_API_KEY=your_stream_key
STREAM_API_SECRET=your_stream_secret

# Other services
GEMINI_API_KEY=your_gemini_key
CARTESIA_API_KEY=your_cartesia_key
DEEPGRAM_API_KEY=your_deepgram_key
```

## Usage

1. Run the script: `uv run python main.py`
2. A browser window will open with the video call
3. Enable your camera
4. Show objects to the camera
5. The video will display bounding boxes around detected objects
6. Ask the agent: "What do you see?"
7. The agent will describe the detected objects

## Rate Limits

Roboflow's hosted API has rate limits. This example uses 5 FPS by default to be API-friendly.

For higher throughput:
- Request a higher rate limit from Roboflow
- Use a dedicated Roboflow inference server
- Reduce FPS if you hit limits

## Customization

### Change Detection Model

Update the environment variables to point to a different Roboflow project:

```bash
ROBOFLOW_WORKSPACE=different-workspace
ROBOFLOW_PROJECT=different-project
ROBOFLOW_VERSION=2
```

### Adjust Confidence Threshold

Lower values detect more objects but may include false positives:

```python
conf_threshold=30,  # More sensitive (30%)
```

Higher values are more strict:

```python
conf_threshold=60,  # Less sensitive (60%)
```

### Change Frame Rate

Process more or fewer frames per second:

```python
fps=10,  # Higher rate (check your API limits)
fps=2,   # Lower rate (more conservative)
```

## Troubleshooting

### "API key required" Error

Make sure `ROBOFLOW_API_KEY` is set in your `.env` file.

### "Missing Roboflow configuration" Error

Verify `ROBOFLOW_WORKSPACE` and `ROBOFLOW_PROJECT` are set correctly.

### Rate Limit Errors

Reduce the `fps` parameter or request a higher rate limit from Roboflow.

### No Detections

- Check your model is trained and deployed in Roboflow
- Lower the `conf_threshold` value
- Verify the objects you're showing match your model's training

## Learn More

- [Roboflow Documentation](https://docs.roboflow.com/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [Stream Video Documentation](https://getstream.io/video/docs/)

