# Turn Detection Implementation

This directory contains turn detection implementations for Stream agents, including integration with FAL AI's smart-turn model.

## FAL Smart-Turn Integration

The `FalTurnDetection` class provides integration with [FAL AI's smart-turn model](https://fal.ai/models/fal-ai/smart-turn) to detect when speakers complete their turns in conversations.

### Features

- **Real-time Audio Processing**: Buffers and processes audio from Stream WebRTC calls
- **Smart Turn Detection**: Uses FAL AI's model to predict when speakers have finished talking
- **Event-Driven Architecture**: Emits turn events that agents can listen to
- **Configurable Thresholds**: Adjustable confidence thresholds and buffer durations

### Usage

```python
from turn_detection import FalTurnDetection
from agents import Agent

# Create turn detection with custom settings
turn_detection = FalTurnDetection(
    buffer_duration=3.0,        # Process 3 seconds of audio at a time
    prediction_threshold=0.7,   # Confidence threshold for "complete" predictions
    mini_pause_duration=0.5,    # Mini pause detection
    max_pause_duration=2.0      # Max pause detection
)

# Use with an agent
agent = Agent(
    llm=your_llm,
    stt=your_stt,
    tts=your_tts,
    turn_detection=turn_detection,
    name="Turn Detection Bot"
)
```

### Configuration

- `buffer_duration`: How much audio to collect before sending to FAL API (default: 2.0 seconds)
- `prediction_threshold`: Probability threshold for "complete" predictions (default: 0.5)
- `mini_pause_duration`: Duration for mini pause detection (default: 0.5 seconds)
- `max_pause_duration`: Duration for max pause detection (default: 3.0 seconds)

### Environment Variables

- `FAL_KEY`: Your FAL API key (required)

### Events

The `FalTurnDetection` class emits the following events:

- `turn_started`: When a participant starts speaking
- `turn_ended`: When a participant finishes their turn (based on FAL prediction)
- `speech_started`: When speech is detected
- `speech_ended`: When speech ends

### Example

See `examples/example_turn_detection.py` for a complete working example.

## API Response Format

The FAL smart-turn API returns predictions in this format:

```json
{
  "prediction": 1,
  "probability": 0.85,
  "metrics": {
    "inference_time": 0.012,
    "total_time": 0.013
  }
}
```

Where:
- `prediction`: 0 = Incomplete turn, 1 = Complete turn
- `probability`: Confidence score (0.0 - 1.0)
- `metrics`: Performance metrics from the API

## Architecture

```
Audio Input (PcmData)
       ↓
   Audio Buffer
       ↓
   WAV File Creation
       ↓
   FAL API Upload
       ↓
   Smart-Turn Prediction
       ↓
   Event Emission
```

The implementation:
1. Buffers incoming audio data from participants
2. Creates temporary WAV files when enough audio is collected
3. Uploads audio files to FAL's storage
4. Submits audio URLs to the smart-turn model
5. Processes predictions and emits appropriate turn events
6. Cleans up temporary files

## Dependencies

- `fal-client`: For FAL API integration
- `wave`: For audio file creation
- `asyncio`: For asynchronous processing
- `tempfile`: For temporary file management
