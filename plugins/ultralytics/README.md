# Ultralytics Plugin

This plugin provides YOLO-based pose detection capabilities for vision-agents using the Ultralytics YOLO library.

## Features

- Real-time pose detection using YOLO models
- Hand and wrist tracking with detailed connections
- Video processing with pose annotations
- Configurable confidence thresholds and processing intervals
- Support for both CPU and GPU inference

## Installation

```bash
pip install vision-agents-plugins-ultralytics
```

## Usage

```python
from vision_agents.plugins import ultralytics

# Create a YOLO pose processor
processor = ultralytics.YOLOPoseProcessor(
    model_path="yolo11n-pose.pt",
    conf_threshold=0.5,
    device="cpu",
    enable_hand_tracking=True,
    enable_wrist_highlights=True
)
```

## Configuration

- `model_path`: Path to YOLO pose model file (default: "yolo11n-pose.pt")
- `conf_threshold`: Confidence threshold for pose detection (default: 0.5)
- `imgsz`: Image size for YOLO inference (default: 512)
- `device`: Device to run inference on ('cpu' or 'cuda')
- `max_workers`: Number of worker threads for processing (default: 2)
- `interval`: Processing interval in seconds (0 for every frame)
- `enable_hand_tracking`: Whether to draw detailed hand connections
- `enable_wrist_highlights`: Whether to highlight wrist positions

## Dependencies

- ultralytics>=8.0.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- pillow>=10.0.0
- aiortc>=1.6.0
- av>=10.0.0
