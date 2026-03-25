# Roboflow Video Moderation Example

Real-time content moderation that detects offensive gestures using a custom Roboflow model running **locally** via the Roboflow `inference` package. Detected regions are censored with a Gaussian blur in the video stream.

## Setup

1. Install dependencies:

```bash
uv run --directory examples/11_moderation_example moderation_example.py --help
```

2. Create a `.env` file with your API keys:

```bash
cp env.example .env
# Edit .env with your actual credentials
```

3. Update `MODEL_ID` in `moderation_example.py` with your Roboflow model identifier (e.g. `"middle-finger-detection/1"`).

## Running the Example

```bash
# From the repo root:
uv run --directory examples/11_moderation_example moderation_example.py run
```

The agent will:

1. Connect to GetStream and join a video call
2. Download and load the Roboflow model locally (first run only)
3. Process video frames at 5 FPS using local inference
4. Censor any detected offensive gestures with a heavy Gaussian blur
5. Issue verbal warnings via the LLM when gestures are detected

## How It Works

The `LocalModerationProcessor` extends the Roboflow cloud detection processor but replaces cloud inference with local inference:

- Uses `inference.get_model()` to download model weights once and run detection on your machine
- No cloud round-trip per frame — significantly lower latency
- Detected regions are covered with a heavy Gaussian blur so they are not visible to other participants
- A red border is drawn around censored areas for visibility
- Detection events are still emitted so the LLM can issue verbal warnings
- The LLM moderator escalates its tone if gestures persist

## Customization

### Model ID

Replace `MODEL_ID` with any Roboflow model trained for your moderation use case:

```python
MODEL_ID = "your-workspace/your-project/version"
```

### Confidence Threshold

Adjust sensitivity — lower values catch more but may produce false positives:

```python
LocalModerationProcessor(
    model_id=MODEL_ID,
    conf_threshold=0.3,  # More sensitive
    fps=5,
)
```
