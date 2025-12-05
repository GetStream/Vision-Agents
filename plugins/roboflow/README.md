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

# A Roboflow processor for cloud inference and
# pre-trained models from Roboflow Universe https://universe.roboflow.com/.
processor = roboflow.RoboflowCloudDetectionProcessor(
    api_key="your_api_key",  # or set ROBOFLOW_API_KEY env var
    api_url="https://detect.roboflow.com",  # or set ROBOFLOW_API_URL env var
    model_id="football-players-detection-3zvbc/20",
    classes=["player"],
    conf_threshold=0.5,
    fps=3,
)

# You can also use a Roboflow processor with local inference and RF-DETR models.
processor = roboflow.RoboflowLocalDetectionProcessor(
    model_id="rfdetr-seg-preview",
    conf_threshold=0.5,
    classes=["person"],
    fps=3,
    # You can pass a custom model as "model" parameter here.
    # The model must be an instance of `rfdetr.RFDETR()` class.
    # model=MyRF_DETRModel()
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

## RoboflowCloudDetectionProcessor Configuration

- `model_id`: Roboflow Universe model id. Example - `"football-players-detection-3zvbc/20"`.
- `api_key`: Roboflow API key. If not provided, will use `ROBOFLOW_API_KEY` env variable.
- `api_url`: Roboflow API url. If not provided, will use `ROBOFLOW_API_URL` env variable.
- `conf_threshold`: Confidence threshold for detections (0 - 1.0). Default - `0.5`.
- `fps`: Frame processing rate. Default - `5`.
- `classes`: an optional list of class names to be detected.
  Example - `["person", "sports ball"]`
  Verify that the classes a supported by the given model.
  Default - `None` (all classes are detected).
  `annotate`: if True, annotate the detected objects with boxes and labels.
  Default - `True`.
- `dim_background_factor`: how much to dim the background around detected objects from 0 to 1.0.
  Effective only when `annotate=True`.
  Default - `0.0` (no dimming).
- `client`: an optional custom instance of `inference_sdk.InferenceHTTPClient`.

## RoboflowLocalDetectionProcessor Configuration

- `model_id`: identifier of the model to be used.
  Available models are: "rfdetr-base", "rfdetr-large", "rfdetr-nano", "rfdetr-small", "rfdetr-medium", "
  rfdetr-seg-preview".
  Default - `"rfdetr-seg-preview"`.
- `conf_threshold`: Confidence threshold for detections (0 - 1.0). Default - `0.5`.

- `fps`: Frame processing rate. Default - `10`.
- `classes`: optional list of class names to be detected.
  Example: `["person", "sports ball"]`
  Verify that the classes a supported by the given model.
  Default - `None` (all classes are detected).
- `annotate`: if True, annotate the detected objects with boxes and labels.
  Default - True.
- `dim_background_factor`: how much to dim the background around detected objects from 0 to 1.0.
  Effective only when `annotate=True`.
  Default - `0.0` (no dimming).
- `model`: optional instance of `RFDETRModel` to be used for detections.
  Use it provide a model of choosing with custom parameters.

## Testing

```bash
# Run all tests
pytest plugins/roboflow/tests/ -v

# Run specific tests
pytest plugins/roboflow/tests/test_roboflow.py -v
```

## Dependencies

- `vision-agents` - Core framework
- `numpy>=2.0.0` - Array operations
- `rfdetr>=1.3.0` - RF-DETR models for local object detection
- `inference-sdk>=0.26.1` - Roboflow SDK for cloud inference

## Links

- [Roboflow Documentation](https://docs.roboflow.com/)
- [RF-DETR Github](https://github.com/roboflow/rf-detr)
- [Roboflow Inference Documentation](https://inference.roboflow.com/)
- [Vision Agents Documentation](https://visionagents.ai/)
- [GitHub Repository](https://github.com/GetStream/Vision-Agents)

