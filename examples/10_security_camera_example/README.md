# Security Camera Demo

A real-time security camera demo with face recognition that identifies unique visitors, tracks their activity, and provides an AI assistant to answer questions about security.

## Features

- 🎥 **Real-time Face Detection**: Uses face_recognition library for accurate face detection
- 🧠 **Face Recognition**: Identifies unique individuals and prevents duplicate entries
- 👥 **Visitor Tracking**: Maintains a 30-minute sliding window of unique visitors
- 📊 **Visual Overlay**: Displays unique visitor count and face thumbnails in a grid
- 🔢 **Detection Counter**: Shows how many times each visitor has been seen
- 🤖 **AI Integration**: Ask the AI assistant detailed questions about visitor activity
- ⏰ **Timestamp Display**: Shows current date/time on the video feed

## How It Works

### Architecture

The demo uses a custom `SecurityCameraProcessor` that:

1. **Subscribes to Video Stream**: Uses `VideoForwarder` to receive frames from the camera
2. **Detects Faces**: Runs face_recognition on frames at configurable intervals
3. **Generates Face Encodings**: Creates unique 128-dimensional encodings for each face
4. **Matches Faces**: Compares new detections against known faces to identify individuals
5. **Tracks Visitors**: Stores unique visitors with first/last seen timestamps and detection counts
6. **Cleans Old Data**: Automatically removes visitors not seen within the time window
7. **Creates Overlay**: Composites face thumbnails with detection counts onto the video
8. **Publishes Output**: Sends the annotated video to participants via `QueuedVideoTrack`

### Video Overlay

The right side of the video shows:
- **Header**: "SECURITY CAMERA"
- **Visitor Count**: Number of unique visitors in last 30 minutes
- **Face Grid**: Thumbnails of unique visitors (up to 12 most recent)
- **Detection Badges**: Small counters showing how many times each person was seen (e.g., "3x")
- **Timestamp**: Current date and time at bottom of frame

### LLM Integration

The AI assistant has access to:
- `get_visitor_count()`: Get count of unique visitors and total detections
- `get_visitor_details()`: Get detailed info on each visitor (first seen, last seen, detection count)
- Real-time processor state with timing information
- Ability to answer detailed questions about security activity and specific visitors

## Setup

### Prerequisites

- Python 3.13+
- Webcam/camera access
- GetStream account for video transport
- API keys for Gemini, Deepgram, and ElevenLabs

### Installation

1. Navigate to this directory:
```bash
cd examples/10_security_camera_example
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables in `.env`:
```bash
# Stream API credentials
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# LLM API key
GOOGLE_API_KEY=your_gemini_api_key

# STT API key
DEEPGRAM_API_KEY=your_deepgram_api_key

# TTS API key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Usage

### Running the Demo

```bash
uv run python security_camera_example.py
```

The agent will join a call and start monitoring the video feed for faces.

### Interacting with the AI

Once connected, you can ask questions like:
- "How many unique people have visited in the last 30 minutes?"
- "How many times has each person been seen?"
- "When was the last person detected?"
- "Who visited most recently?"
- "Give me details on all visitors"
- "Is the same person still there?"

### Configuration

You can adjust the processor parameters in `security_camera_example.py`:

```python
security_processor = SecurityCameraProcessor(
    fps=5,                      # Frames per second to process
    time_window=1800,           # Time window in seconds (30 min)
    thumbnail_size=80,          # Size of face thumbnails in pixels
    detection_interval=2.0,     # Seconds between face detections
    face_match_tolerance=0.6,   # Face matching tolerance (lower = stricter)
)
```

## Implementation Details

### Face Detection & Recognition

Uses the `face_recognition` library (built on dlib) which:
- Provides state-of-the-art face detection accuracy
- Generates 128-dimensional face encodings for recognition
- Can identify the same person across different angles and lighting
- Uses HOG (Histogram of Oriented Gradients) for detection
- Automatically downloads pre-trained models on first use
- More accurate than Haar Cascades, though slightly slower

### Performance Optimization

- **Threading**: CPU-intensive operations run in ThreadPoolExecutor
- **Configurable FPS**: Process only N frames per second (default: 5)
- **Detection Throttling**: Only detect faces every N seconds (default: 2s)
- **Automatic Cleanup**: Old faces are removed efficiently

### Memory Management

- Face thumbnails are resized to 80x80 pixels to save memory
- Sliding window automatically removes old detections
- Maximum of 12 thumbnails displayed at once

## Extending the Demo

### Add Named Face Recognition

To identify specific people by name (the system currently recognizes but doesn't name individuals):
1. Create a database of known faces with names
2. Generate and store encodings for each known person
3. Compare new detections against known encodings
4. Label faces with names in the overlay instead of showing generic IDs

### Add Motion Detection

To detect movement before face detection:
1. Use OpenCV's background subtraction
2. Only run face detection when motion is detected
3. Reduce false positives and improve performance

### Add Recording

To save clips when faces are detected:
1. Use OpenCV's VideoWriter
2. Buffer frames around detection events
3. Save short clips to disk with timestamps

### Add Alerts

To notify when faces are detected:
1. Check visitor count thresholds
2. Send notifications via email/SMS/webhook
3. Trigger alerts for after-hours detections

## Troubleshooting

### No faces detected
- Ensure good lighting conditions
- Position camera to show frontal faces clearly
- The face_recognition library works best with direct face views
- Try lowering `detection_interval`
- Check that the camera is working properly

### Performance issues
- Reduce `fps` parameter (e.g., from 5 to 2-3)
- Increase `detection_interval` (e.g., from 2 to 3-5 seconds)
- Lower video resolution
- Use fewer `max_workers`
- Face recognition is more CPU-intensive than simple detection

### Too many duplicate faces
- Decrease `face_match_tolerance` (e.g., 0.5 instead of 0.6)
- Increase `detection_interval` to give more time between checks

### Same person not being recognized
- Increase `face_match_tolerance` (e.g., 0.7 instead of 0.6)
- Ensure consistent lighting and camera angle
- Face encodings work better with clear, frontal face views

### Video lag
- Check network bandwidth
- Reduce frame processing rate
- Ensure camera has good connection

## License

See the main repository LICENSE file for details.

