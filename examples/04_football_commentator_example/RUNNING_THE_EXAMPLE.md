# Football Commentator Example

A real-time AI sports commentator that watches football footage and provides play-by-play commentary using OpenAI Realtime and Roboflow object detection.

## Setup

1. **Get API keys:**
   - OpenAI API key: https://platform.openai.com/api-keys
   - GetStream API key: https://getstream.io

2. **Configure environment:**
   ```bash
   cd examples/03_football_commentator_example
   cp env.example .env
   # Edit .env with your actual API keys
   ```

3. **Run the example:**
   ```bash
   uv run football_commentator_example.py
   ```

## What It Does

The agent:

- Joins a video call with real-time object detection
- Detects players and the ball using RF-DETR (runs locally, no Roboflow API key needed)
- Annotates the video feed with bounding boxes
- When the ball is detected, prompts OpenAI Realtime to describe the action
- Uses debouncing (8s) to avoid overwhelming the model with requests

## How to Test

1. Run the example and open the demo UI link in the console
2. Join the call
3. Share your screen with football footage playing (or use `--video-track-override` with a local video file)
4. Watch the annotated video and listen to the AI commentary

To use a local video file instead of screen sharing:
```bash
uv run football_commentator_example.py --video-track-override path/to/football.mp4
```

## Configuration

Edit `football_commentator_example.py` to customize:

```python
roboflow.RoboflowLocalDetectionProcessor(
    classes=["person", "sports ball"],  # Objects to detect
    conf_threshold=0.5,                  # Detection confidence (0-1)
    fps=5,                               # Detection frame rate
)
```

The commentary prompts are in the `questions` list. The system instructions are in `instructions.md`.
