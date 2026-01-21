# Security Camera Demo

A real-time security camera demo with face recognition, package detection, and automated package theft response -
including wanted poster generation and posting to X.

## Features

- 🎥 **Real-time Face Detection**: Uses [face_recognition](https://github.com/ageitgey/face_recognition) library for
  accurate face detection and recognition
- 📦 **Package Detection**: [YOLOv11](https://docs.ultralytics.com/models/yolo11/)-based object detection for packages
  and boxes
- 🚨 **Package Theft Detection**: Detects when a package disappears and identifies the suspect
- 🖼️ **Wanted Poster Generation**: Automatically creates a wanted poster when a package is "stolen"
- 🐦 **X Integration**: Posts wanted posters to [X](https://developer.x.com) automatically
- 👤 **Named Face Recognition**: Remembers people by name ("remember me as xxx")
- 📊 **Visual Overlay**: Displays visitor count, package count, and thumbnail grid
- 📜 **Activity Log**: Tracks events like arrivals, departures, and package detections
- 🤖 **AI Integration**: Ask the AI assistant questions about security activity

## How It Works

### Architecture

The demo uses a custom `SecurityCameraProcessor` that:

1. **Subscribes to Video Stream**: Uses `VideoForwarder` to receive frames from the camera
2. **Detects Faces**: Runs face_recognition on frames at configurable intervals
3. **Detects Packages**: Runs YOLO model to detect packages, boxes, and parcels
4. **Matches Faces**: Compares new detections against known faces to identify individuals
5. **Tracks Visitors & Packages**: Stores unique visitors and packages with timestamps
6. **Detects Theft**: When a package disappears while someone is present, triggers theft workflow
7. **Creates Overlay**: Composites face and package thumbnails onto the video
8. **Publishes Output**: Sends the annotated video to participants via `QueuedVideoTrack`

### Package Theft Workflow

When a package disappears from the frame:

1. System identifies who was present when the package disappeared
2. Waits 3 seconds to confirm the package is truly gone (not just a detection blip)
3. Generates a "WANTED" poster with the suspect's face
4. Displays the poster in the video call for 8 seconds
5. Posts the poster to X with a caption
6. Agent announces the theft and poster generation

### Video Overlay

The right side of the video shows:

- **Header**: "SECURITY CAMERA"
- **Visitor Count**: Currently visible / total unique visitors
- **Package Count**: Currently visible / total packages seen
- **Legend**: Color coding for people (green) and packages (blue)
- **Thumbnail Grid**: Up to 12 most recent faces and packages
- **Detection Badges**: Show how many times each person/package was seen
- **Timestamp**: Current date and time at bottom of frame

Bounding boxes are drawn around detected faces (green) and packages (blue).

### LLM Integration

The AI assistant has access to:

- `get_visitor_count()`: Get count of unique visitors and total detections
- `get_visitor_details()`: Get detailed info on each visitor (first seen, last seen, detection count)
- `get_package_count()`: Get current and total package counts
- `get_package_details()`: Get history of all packages including who picked them up
- `get_activity_log()`: Get recent events (arrivals, packages detected, departures)
- `remember_my_face(name)`: Register a face so it's recognized by name in the future
- `get_known_faces()`: List all registered faces

## Setup

### Prerequisites

- Python 3.13+
- Webcam/camera access
- [GetStream](https://getstream.io) account for video transport
- API keys for [Gemini](https://ai.google.dev), [Deepgram](https://deepgram.com),
  and [ElevenLabs](https://elevenlabs.io)
- (Optional) [X Developer API](https://developer.x.com) credentials for posting wanted posters

### Installation

1. Navigate to this directory:

```bash
cd examples/04_security_camera_example
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

# X (Twitter) API credentials (optional, for posting wanted posters)
X_API_KEY=your_x_api_key
X_API_SECRET=your_x_api_secret
X_ACCESS_TOKEN=your_x_access_token
X_ACCESS_TOKEN_SECRET=your_x_access_token_secret
```

## Usage

### Running the Demo

```bash
uv run security_camera_example.py run
```

The agent will join a call and start monitoring the video feed for faces and packages.

### Interacting with the AI

Once connected, you can ask say things like:

- "How many people have visited?"
- "What happened while I was away?"
- "Did anyone come by?"
- "Have any packages been delivered?"
- "Who picked up the package?"
- "Remember me as xxx"
- "Who do you know?"

### Package Theft Demo

To trigger the theft workflow:

1. Place a package (box, parcel) in view of the camera
2. Wait for it to be detected (blue bounding box appears)
3. Have someone pick up the package while their face is visible
4. The system will generate a wanted poster and post it to X

### Configuration

You can adjust the processor parameters in `security_camera_example.py`:

```python
security_processor = SecurityCameraProcessor(
    fps=5,  # Frames per second to process
    time_window=1800,  # Time window in seconds (30 min)
    thumbnail_size=80,  # Size of thumbnails in pixels
    detection_interval=2.0,  # Seconds between face detection with identity matching
    bbox_update_interval=0.3,  # Seconds between fast bbox updates for tracking
    model_path="weights_custom.pt",  # YOLO model for package detection
    package_conf_threshold=0.7,  # Package detection confidence threshold
    max_tracked_packages=1,  # Single-package mode for demo
    face_match_tolerance=0.6,  # Face matching tolerance (lower = stricter)
)
```

## Implementation Details

### Face Detection & Recognition

Uses the [face_recognition](https://github.com/ageitgey/face_recognition) library (built on dlib) which:

- Provides state-of-the-art face detection accuracy
- Generates 128-dimensional face encodings for recognition
- Can identify the same person across different angles and lighting
- Supports named face registration for persistent recognition

### Package Detection

Uses a custom [YOLOv11](https://docs.ultralytics.com/models/yolo11/) model (`weights_custom.pt`) trained to detect:

- Box
- Box_broken
- Open_package
- Package

The model runs package detection at configurable intervals with IoU-based tracking to maintain package identity across
frames.

### About the Custom Model

The `weights_custom.pt` file is a YOLOv11 object detection model we trained using [Roboflow](https://roboflow.com)
with [SAM 3](https://blog.roboflow.com/sam3/) for assisted labeling. SAM 3's text-prompt segmentation made it fast to
annotate packages and boxes accurately.

**We are not distributing `weights_custom.pt`.** To run this demo, you'll need to provide your own YOLO model.

Options:

- **Train your own**: Use [Roboflow](https://roboflow.com) to label a dataset and train a YOLOv11 model. See
  their [YOLOv11 training guide](https://blog.roboflow.com/yolov11-how-to-train-custom-data/).
- **Find a pre-trained model**: Search [Roboflow Universe](https://universe.roboflow.com) for "package detection"
  datasets and models.

Place your model weights at `weights_custom.pt` in this directory, or change the `model_path` parameter.

### Wanted Poster Generation

The `poster_generator.py` module:

- Creates a stylized "WANTED" poster with the suspect's face
- Uses PIL/Pillow for image composition
- Optionally posts to X using the Twitter API v2

### Event System

The processor emits events that the agent subscribes to:

- `PersonDetectedEvent`: New or returning person detected
- `PersonDisappearedEvent`: Person left the frame
- `PackageDetectedEvent`: New or returning package detected
- `PackageDisappearedEvent`: Package disappeared (potential theft)

## External Resources

- [face_recognition](https://github.com/ageitgey/face_recognition): Python face recognition library
- [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/): Object detection model
- [Roboflow](https://roboflow.com): Dataset management, labeling, and model training
- [SAM 3 (Segment Anything 3)](https://blog.roboflow.com/sam3/): Foundation model for assisted labeling
- [GetStream](https://getstream.io): Video transport infrastructure
- [Deepgram](https://deepgram.com): Speech-to-text API
- [ElevenLabs](https://elevenlabs.io): Text-to-speech API
- [Google Gemini](https://ai.google.dev): LLM API
- [X Developer Portal](https://developer.x.com): Twitter/X API

## License

See the main repository LICENSE file for details.
