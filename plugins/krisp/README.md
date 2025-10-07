# Krisp Plugin

This plugin provides Krisp-based turn detection capabilities for Vision Agents using the Krisp Audio SDK.

## Features

- **Turn Detection**: Real-time detection of speaking turns using Krisp's turn-taking model
- **High Accuracy**: Leverages Krisp's advanced audio processing for reliable turn detection
- **Configurable**: Adjustable confidence thresholds and frame durations
- **Low Latency**: Optimized for real-time conversational AI applications

## Installation

```bash
pip install vision-agents-plugins-krisp
```

**Note**: This plugin requires the Krisp Audio SDK and a valid Krisp model file (`.kef`).

## Usage

```python
from vision_agents.plugins import krisp

# Create turn detector
turn_detector = krisp.TurnDetection(
    model_path="path/to/krisp-viva-tt-v1.kef",
    frame_duration_ms=15,
    confidence_threshold=0.5
)

# Start detection
turn_detector.start()

# Process audio
await turn_detector.process_audio(audio_data, user_id="user123")

# Stop detection
turn_detector.stop()
```

## Configuration

- `model_path`: Path to the Krisp model file (`.kef`)
- `frame_duration_ms`: Duration of audio frames in milliseconds (10, 15, 20, 30, or 32)
- `confidence_threshold`: Threshold for turn detection (0.0 to 1.0)

## Requirements

- Python 3.10+
- `krisp-audio` SDK
- `vision-agents-core`
- Krisp model file

## License

Apache-2.0

