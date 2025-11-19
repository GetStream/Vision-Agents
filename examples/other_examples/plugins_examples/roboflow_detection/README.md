# Roboflow Object Detection Example

This example demonstrates real-time object detection in video calls using the Roboflow plugin.

## What It Does

- Joins a Stream video call as an AI agent
- Processes video frames at 5 FPS using a Roboflow model
- Detects objects based on the model's training
- Annotates video with bounding boxes and labels
- Describes what it sees when you ask

## Quick Start (5 Minutes)

### 1. Get Your Roboflow API Key

1. Sign up at https://app.roboflow.com (free)
2. Go to **Settings** → **Roboflow API**
3. Copy your **Private API Key**

### 2. Find a Model on Universe

1. Browse models at https://universe.roboflow.com
2. Example: Search for "mobile phone detection"
3. Click on a model you like (e.g., https://universe.roboflow.com/tusker-ai/mobile-phone-detection-2vads)
4. Note the URL structure: `universe.roboflow.com/{workspace}/{project}`
5. Your **model_id** is: `{workspace}/{project}` (e.g., `tusker-ai/mobile-phone-detection-2vads`)

**Popular Universe Models:**
- `tusker-ai/mobile-phone-detection-2vads` - Mobile phone detection
- `roboflow-100/aerial-spheres` - Aerial objects
- `roboflow-100/license-plates` - License plates
- `microsoft/coco` - 80 common objects (person, car, etc.)

### 3. Test the Model (Recommended)

```bash
cd examples/other_examples/plugins_examples/roboflow_detection
export ROBOFLOW_API_KEY="your_api_key_here"
uv run python test_model.py
```

This verifies your API key works and the model is accessible.

### 4. Run the Full Example

```bash
# Set up environment
cp env.example .env
# Edit .env with your API keys

# Install dependencies
uv sync

# Run the agent
uv run python main.py
```

A browser window will open with the video call. Enable your camera, show objects, and ask "What do you see?"

## Prerequisites

1. **Roboflow API Key** (required)
   - Get from https://app.roboflow.com → Settings → Roboflow API
   - Works for both private models and public Universe models

2. **Other API Keys** (for the full agent)
   - GetStream API key (https://getstream.io)
   - Gemini API key (https://ai.google.dev)
   - Cartesia API key (https://cartesia.ai)
   - Deepgram API key (https://deepgram.com)

## Important Notes

**Do I Need an API Key for Public Universe Models?**

Yes! Even though Universe models are "public" (anyone can use them), Roboflow still requires an API key for authentication and rate limiting. The good news:
- Any free Roboflow account API key works
- You don't need to own or create the model
- You can use thousands of pre-trained Universe models instantly

**No Pipeline Creation Needed**

If you see prompts about "creating a pipeline" on the Roboflow website, that's for their Workflows feature (chaining multiple models). You can ignore it - this Python SDK approach runs inference directly without any pipeline setup.

## Setup

1. **Install dependencies:**
   ```bash
   cd examples/other_examples/plugins_examples/roboflow_detection
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

3. **Run the example:**
   ```bash
   uv run python main.py
   ```

## Configuration

### Using Your Own Model (Private)

Latest version (automatic):
```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    model_id="your-project-name",
    api_key="your-api-key",  # or set ROBOFLOW_API_KEY env var
    conf_threshold=40,
    fps=5,
)
```

Specific version:
```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    model_id="your-project-name",
    version=1,  # specify version
    api_key="your-api-key",
    conf_threshold=40,
    fps=5,
)
```

### Using a Roboflow Universe Model (Public)

Latest version (automatic):
```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    model_id="roboflow-100/aerial-spheres",
    conf_threshold=40,
    fps=5,
)
```

Specific version in model_id:
```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    model_id="roboflow-100/aerial-spheres/2",
    conf_threshold=40,
    fps=5,
)
```

Specific version as parameter:
```python
roboflow_processor = roboflow.RoboflowDetectionProcessor(
    model_id="roboflow-100/aerial-spheres",
    version=2,
    conf_threshold=40,
    fps=5,
)
```

### Environment Variables

Required variables in `.env`:

```bash
# Roboflow (only needed for private models)
ROBOFLOW_API_KEY=your_api_key

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

Simply change the `model_id` parameter:

```python
# Use a different Universe model
model_id="microsoft/coco/3"

# Or your own private model
model_id="my-other-project"
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

## Finding Models on Roboflow Universe

1. Browse models at https://universe.roboflow.com
2. Find a model you like (e.g., "Aerial Spheres")
3. Note the workspace and project name from the URL
4. Format: `workspace/project` or `workspace/project/version`

## Troubleshooting

### "API key required" Error

This happens when using a private model. Either:
- Set `ROBOFLOW_API_KEY` in your `.env` file
- Pass `api_key` parameter to the processor
- Use a public Universe model instead (no API key needed)

### Version Not Found

If you get errors about versions:
- Verify your model has at least one trained version in Roboflow
- Check the version number exists (versions start at 1)
- Omit the `version` parameter to automatically use the latest

### Rate Limit Errors

Reduce the `fps` parameter or request a higher rate limit from Roboflow.

### No Detections

- Check your model is trained and deployed in Roboflow
- Lower the `conf_threshold` value
- Verify the objects you're showing match your model's training

## Learn More

- [Roboflow Documentation](https://docs.roboflow.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [Stream Video Documentation](https://getstream.io/video/docs/)

