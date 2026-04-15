# Roboflow Video Moderation Example

Real-time content moderation for video calls. A custom Roboflow model running **locally** detects offensive gestures, censors them with a Gaussian blur, and the AI moderator issues escalating verbal warnings — ultimately kicking the user from the call on the third offence.

## What it does

1. Connects to Stream's edge network and joins a video call
2. Downloads and loads a Roboflow detection model locally (first run only)
3. Processes video frames at 15 FPS using local inference
4. Censors detected gestures with a heavy Gaussian blur
5. Issues verbal warnings via an LLM (Gemini Flash Lite) + TTS (Deepgram)
6. Escalates: gentle warning → stern warning with removal threat → kicks the user via the Stream API

## Setup

1. Install dependencies:

```bash
uv sync --directory examples/11_moderation_example
```

2. Create a `.env` file with your API keys:

```bash
cd examples/11_moderation_example
cp env.example .env
# Edit .env with your actual credentials
```

You'll need:
- `GOOGLE_API_KEY` — for Gemini LLM
- `DEEPGRAM_API_KEY` — for STT and TTS
- `ROBOFLOW_API_KEY` — for downloading model weights
- `STREAM_API_KEY` and `STREAM_API_SECRET` — for the video call transport

3. Update `MODEL_ID` in `moderation_example.py` with your Roboflow model (see below).

## Running

```bash
uv run --directory examples/11_moderation_example moderation_example.py run
```

A browser window will open with the demo UI. Join the call and the agent will start moderating.

## Using your own Roboflow model

The example ships with a model ID for a specific gesture detection model. To use your own:

1. Train or find a detection model on [Roboflow](https://roboflow.com)
2. Copy the model ID from your Roboflow project. The format depends on the model type:
   - Standard models: `"your-workspace/your-project/1"` (workspace/project/version)
   - NAS models: `"your-workspace/your-project-name-hash"` (no version number)
3. Replace `MODEL_ID` in `moderation_example.py`:

```python
MODEL_ID = "your-workspace/your-project/1"
```

The model is downloaded automatically on first run via the `inference` package. Any Roboflow object detection model works — the processor will censor whatever it detects and emit events so the LLM can respond.

### Adjusting detection sensitivity

Lower the confidence threshold to catch more detections (at the cost of more false positives):

```python
LocalModerationProcessor(
    model_id=MODEL_ID,
    conf_threshold=0.3,  # default is 0.4
    fps=15,
)
```

## How it works

The `LocalModerationProcessor` extends the Roboflow cloud detection processor but replaces cloud inference with local inference:

- Uses `inference.get_model()` to download model weights once and run detection on your machine
- Implements `Warmable` so the model loads during agent startup, not on the first frame
- No cloud round-trip per frame — lower latency
- Detected regions are covered with a heavy Gaussian blur
- Detection events trigger LLM responses via `agent.simple_response()`
- A warning lock prevents overlapping warnings; a `_wait_for_tts_playback` helper ensures the full warning is heard before any kick
