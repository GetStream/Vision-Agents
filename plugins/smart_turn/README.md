# Smart Turn Detection Plugin

An AI-powered turn detection plugin for Vision Agents that uses the [Smart Turn model](https://github.com/pipecat-ai/smart-turn) to detect when a speaker has completed their turn in a conversation.

## Overview

Smart Turn is an open-source, community-driven, native audio turn detection model that goes beyond simple Voice Activity Detection (VAD). It analyzes grammar, tone, pace of speech, and various other complex audio and semantic cues to determine when a user has finished speaking, matching human expectations more closely than VAD-based approaches.

For more information about the Smart Turn model, visit the [official repository](https://github.com/pipecat-ai/smart-turn).

## Installation

```bash
pip install vision-agents-plugins-smart-turn
```

## Usage

```python
from vision_agents.plugins.smart_turn import TurnDetection

# Initialize with FAL API key from environment variable
turn_detector = TurnDetection()

# Or specify API key directly
turn_detector = TurnDetection(api_key="your_fal_api_key")

# Register event handlers
@turn_detector.on("turn_started")
def on_turn_started(event_data):
    print(f"Turn started: {event_data.speaker_id}")

@turn_detector.on("turn_ended")
def on_turn_ended(event_data):
    print(f"Turn ended: {event_data.speaker_id} (confidence: {event_data.confidence:.3f})")

# Start detection
turn_detector.start()

# Process audio
await turn_detector.process_audio(pcm_data, user_id="user123")

# Stop detection
turn_detector.stop()
```

## Configuration Options

- `api_key`: FAL API key (default: reads from FAL_KEY environment variable)
- `buffer_duration`: Duration in seconds to buffer audio before processing (default: 2.0)
- `confidence_threshold`: Probability threshold for "complete" predictions (default: 0.5)
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `channels`: Number of audio channels (default: 1)
