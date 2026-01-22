# Vision Agents WebRTC Integration - API Reference

**Version:** 1.0
**Status:** Public API Frozen (Breaking changes require major version bump)
**Last Updated:** 2026-01-22

This document defines the stable public API for integrating external RTC systems with Vision Agents. All APIs documented here are considered stable and follow semantic versioning.

## Table of Contents

1. [EdgeTransport Interface](#edgetransport-interface)
2. [LocalEdge Implementation](#localedge-implementation)
3. [Calling Conventions](#calling-conventions)
4. [Audio Format Negotiation](#audio-format-negotiation)
5. [Track APIs](#track-apis)
6. [Room Protocol](#room-protocol)
7. [Best Practices](#best-practices)
8. [Deprecation Policy](#deprecation-policy)

---

## EdgeTransport Interface

The `EdgeTransport` abstract base class defines the interface that all RTC transport implementations must follow.

**Location:** `vision_agents.core.edge.EdgeTransport`

### Required Methods

All EdgeTransport implementations **must** implement these methods:

#### `async def create_user(user: User) -> None`

Initialize a user in the transport layer.

**Parameters:**
- `user` (User): User object containing `id`, `name`, and optional `image`

**Example:**
```python
edge = localrtc.Edge()
user = User(id="user-123", name="Alice")
await edge.create_user(user)
```

---

#### `def create_audio_track(**kwargs) -> OutputAudioTrack`

Create an audio output track for playing audio to speakers.

**Parameters:**
- `framerate` (int, optional): Sample rate in Hz. Default varies by implementation.
- `stereo` (bool, optional): Whether to use stereo (True) or mono (False).
- Additional kwargs may be implementation-specific.

**Returns:**
- `OutputAudioTrack`: An audio track instance implementing the OutputAudioTrack protocol.

**Example:**
```python
edge = localrtc.Edge()
audio_track = edge.create_audio_track(framerate=24000, stereo=False)
await audio_track.write(pcm_data)
```

**Notes:**
- Output format is automatically negotiated when `join()` is called with an agent
- For LocalEdge, defaults to 24kHz mono (Gemini's native format)
- For StreamEdge, defaults to 48kHz stereo (WebRTC standard)

---

#### `async def create_call(call_type: str, call_id: str) -> Room`

Create a new call/room.

**Parameters:**
- `call_type` (str): Type of call (e.g., "default", "audio", "video")
- `call_id` (str): Unique identifier for the call

**Returns:**
- `Room`: A Room instance with `id`, `type` properties and `leave()` method

**Example:**
```python
edge = localrtc.Edge()
room = await edge.create_call("default", "my-call-123")
print(f"Created room {room.id} of type {room.type}")
```

---

#### `async def join(*args, **kwargs) -> Room`

Join a room and start device capture/streaming.

**Recommended Calling Convention:**
```python
room = await edge.join(agent, room_id="call-1", room_type="default")
```

**Legacy Calling Convention (Supported):**
```python
room = await edge.join(room_id="call-1", room_type="default")
```

**Parameters:**
- `*args`: First positional argument should be Agent instance (recommended for audio negotiation)
- `room_id` (str, kwarg): Unique identifier for the room
- `room_type` (str, kwarg): Type of room (default: "default")

**Returns:**
- `Room`: The joined room instance

**Example:**
```python
edge = localrtc.Edge()
agent = Agent(edge=edge, llm=gemini.Realtime())
room = await edge.join(agent, room_id="my-call", room_type="default")
```

**Notes:**
- Passing an agent enables automatic audio format negotiation
- Without an agent, output format defaults to 24kHz mono

---

#### `async def publish_tracks(room: Room, audio_track=None, video_track=None) -> None`

Publish media tracks to the room.

**Recommended Calling Convention:**
```python
await edge.publish_tracks(room, audio_track=audio, video_track=video)
```

**Parameters:**
- `room` (Room): The room to publish tracks to
- `audio_track` (optional): Audio track instance (e.g., AudioInputTrack)
- `video_track` (optional): Video track instance (e.g., VideoInputTrack)

**Example:**
```python
room = await edge.join(agent, room_id="my-call")
audio = AudioInputTrack(device="default", sample_rate=16000)
video = VideoInputTrack(device=0)
await edge.publish_tracks(room, audio_track=audio, video_track=video)
```

**Deprecation Notice:**
- Legacy convention without explicit room parameter is deprecated
- Will be removed in version 2.0

---

#### `async def close() -> None`

Close the transport and clean up resources.

**Example:**
```python
edge = localrtc.Edge()
# ... use edge ...
await edge.close()  # Clean up when done
```

**Notes:**
- Stops all audio/video capture
- Closes network connections
- Releases device resources
- Should always be called when transport is no longer needed

---

### Optional Methods

These methods may be implemented as no-ops if not applicable:

#### `def open_demo(*args, **kwargs) -> None`

Open a demo session (web UI, browser, etc.).

**Notes:**
- LocalEdge implements this as a no-op
- StreamEdge opens GetStream's demo UI

---

#### `async def create_conversation(call: Any, user: User, instructions: Any) -> None`

Create a conversation in the call.

**Notes:**
- LocalEdge implements this as a no-op (conversations managed by LLM)
- StreamEdge creates conversation in GetStream service

---

#### `def add_track_subscriber(track_id: str, callback: Callable[[PcmData], None]) -> Optional[MediaStreamTrack]`

Subscribe to a track and receive data via callback.

**Parameters:**
- `track_id` (str): Track identifier (e.g., "audio", "video")
- `callback` (Callable): Function to receive PcmData

**Returns:**
- `Optional[MediaStreamTrack]`: The subscribed track, or None

**Example:**
```python
def on_audio(pcm_data: PcmData):
    print(f"Received audio: {len(pcm_data.data)} bytes")

track = edge.add_track_subscriber("audio", on_audio)
```

---

## LocalEdge Implementation

The `LocalEdge` class implements EdgeTransport for local device access.

**Location:** `vision_agents.plugins.localrtc.Edge`

### Constructor

```python
def __init__(
    audio_device: Union[str, int] = "default",
    video_device: Union[int, str] = 0,
    speaker_device: Union[str, int] = "default",
    sample_rate: int = 16000,
    channels: int = 1,
    custom_pipeline: Optional[Dict[str, Any]] = None,
) -> None
```

**Parameters:**
- `audio_device`: Audio input device identifier (default: "default")
  - Can be device name (str) or index (int)
- `video_device`: Video input device identifier (default: 0)
  - Can be device index (int) or path (str)
- `speaker_device`: Audio output device identifier (default: "default")
  - Can be device name (str) or index (int)
- `sample_rate`: Audio capture sample rate in Hz (default: 16000)
  - Common values: 8000, 16000, 44100, 48000
- `channels`: Number of audio channels (default: 1)
  - 1 = mono, 2 = stereo
- `custom_pipeline`: Optional GStreamer pipeline configuration (dict)
  - Keys: `audio_source`, `video_source`, `audio_sink`
  - Values: GStreamer pipeline strings

**Example:**
```python
# Basic usage with defaults
edge = localrtc.Edge()

# Custom device selection
edge = localrtc.Edge(
    audio_device="MacBook Pro Microphone",
    video_device=0,
    speaker_device="External Speakers",
    sample_rate=16000,
    channels=1,
)

# GStreamer pipeline for advanced control
pipeline = {
    "audio_source": "alsasrc device=hw:0 ! audioconvert ! audioresample",
    "video_source": "v4l2src device=/dev/video0 ! videoconvert",
    "audio_sink": "alsasink device=hw:0"
}
edge = localrtc.Edge(custom_pipeline=pipeline)
```

---

### Static Methods

#### `Edge.list_devices() -> Dict[str, List[DeviceInfo]]`

Enumerate available audio and video devices.

**Returns:**
- Dictionary with keys:
  - `audio_inputs`: List of audio input devices
  - `audio_outputs`: List of audio output devices
  - `video_inputs`: List of video input devices
- Each device is a dict with `name` and `index` keys

**Example:**
```python
devices = localrtc.Edge.list_devices()
print("Audio inputs:")
for device in devices["audio_inputs"]:
    print(f"  {device['index']}: {device['name']}")

# Use device by index
edge = localrtc.Edge(audio_device=devices["audio_inputs"][0]["index"])
```

---

## Calling Conventions

### Standard Convention (Recommended)

All public API methods use **keyword arguments** for clarity:

```python
# ✅ RECOMMENDED
edge = localrtc.Edge(
    audio_device="default",
    video_device=0,
    sample_rate=16000,
    channels=1,
)

room = await edge.join(agent, room_id="call-1", room_type="default")

await edge.publish_tracks(
    room,
    audio_track=audio,
    video_track=video,
)
```

### Legacy Conventions (Deprecated)

Some methods support legacy positional argument patterns for backward compatibility:

```python
# ⚠️ DEPRECATED - Will be removed in v2.0
await edge.publish_tracks(audio, video)  # Missing room parameter
```

**Deprecation Warnings:**
- Legacy conventions issue `DeprecationWarning` at runtime
- Update code to use keyword arguments with explicit parameters
- Legacy support will be removed in version 2.0

---

## Audio Format Negotiation

### Overview

Vision Agents automatically negotiates audio formats between capture devices and LLM providers to ensure optimal quality without manual resampling.

### How It Works

1. **Input Configuration**: Set via EdgeTransport constructor
   ```python
   edge = localrtc.Edge(sample_rate=16000, channels=1)
   ```

2. **Output Negotiation**: Automatic when agent is passed to `join()`
   ```python
   room = await edge.join(agent, room_id="call-1")
   ```

3. **Format Detection**: Edge queries LLM's `get_audio_requirements()`
   ```python
   # Gemini returns: AudioCapabilities(sample_rate=24000, channels=1, ...)
   ```

4. **Automatic Configuration**: Output track configured to match
   ```python
   # LocalEdge creates AudioOutputTrack with 24kHz mono
   audio_track = edge.create_audio_track()
   ```

### Supported Providers

| Provider | Sample Rate | Channels | Bit Depth | Encoding |
|----------|-------------|----------|-----------|----------|
| Gemini Realtime | 24000 Hz | 1 (mono) | 16-bit | PCM |
| OpenAI Realtime | 24000 Hz | 1 (mono) | 16-bit | PCM |
| Custom | Varies | Varies | Varies | Varies |

### Best Practices

1. **Always pass agent to join()** for automatic negotiation
   ```python
   room = await edge.join(agent, room_id="call-1")  # ✅
   ```

2. **Use 16kHz mono for voice input** (optimal for speech recognition)
   ```python
   edge = localrtc.Edge(sample_rate=16000, channels=1)
   ```

3. **Let framework handle output format** (avoid manual resampling)
   ```python
   audio_track = edge.create_audio_track()  # Automatically configured
   ```

4. **Check logs for format mismatch warnings**
   ```python
   # [AUDIO FORMAT MISMATCH] Input format 44100Hz, 2ch is not directly supported
   ```

---

## Track APIs

### AudioInputTrack

Captures audio from microphone or audio input device.

**Location:** `vision_agents.plugins.localrtc.AudioInputTrack`

```python
class AudioInputTrack:
    def __init__(
        self,
        device: Union[str, int] = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
        buffer_duration: float = 2.0,
    ) -> None

    def start() -> None
    def stop() -> None
    def capture(duration: float) -> PcmData
```

**Example:**
```python
audio = AudioInputTrack(
    device="default",
    sample_rate=16000,
    channels=1,
)
audio.start()
pcm_data = audio.capture(duration=0.1)  # 100ms chunk
audio.stop()
```

---

### AudioOutputTrack

Plays audio to speakers or audio output device.

**Location:** `vision_agents.plugins.localrtc.AudioOutputTrack`

```python
class AudioOutputTrack:
    def __init__(
        self,
        device: Union[str, int] = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
        buffer_size_ms: int = 10000,
    ) -> None

    async def write(data: PcmData) -> None
    async def flush() -> None
    def stop() -> None
```

**Features:**
- Automatic resampling (sample rate conversion)
- Channel mixing (mono ↔ stereo)
- Bit depth conversion

**Example:**
```python
audio = AudioOutputTrack(
    device="default",
    sample_rate=24000,
    channels=1,
)
await audio.write(pcm_data)
await audio.flush()
audio.stop()
```

---

### VideoInputTrack

Captures video from camera or video input device.

**Location:** `vision_agents.plugins.localrtc.VideoInputTrack`

```python
class VideoInputTrack:
    def __init__(
        self,
        device: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None

    def start(callback: Callable[[np.ndarray, float], None]) -> None
    def stop() -> None
```

**Example:**
```python
def on_frame(frame: np.ndarray, timestamp: float):
    print(f"Received frame: {frame.shape}")

video = VideoInputTrack(device=0, width=640, height=480, fps=30)
video.start(on_frame)
# ... process frames ...
video.stop()
```

---

## Room Protocol

The `Room` protocol defines the interface for call/room objects.

**Location:** `vision_agents.core.protocols.Room`

```python
class Room(Protocol):
    @property
    def id(self) -> str
        """Unique identifier for the room."""

    @property
    def type(self) -> str
        """Type of room (e.g., 'default', 'audio', 'video')."""

    async def leave(self) -> None
        """Leave the room and stop participation."""
```

**Example:**
```python
room = await edge.create_call("default", "my-call-123")
print(f"Room ID: {room.id}")
print(f"Room Type: {room.type}")
await room.leave()
```

---

## Best Practices

### 1. Edge Configuration

```python
# ✅ Use explicit keyword arguments
edge = localrtc.Edge(
    audio_device="default",
    video_device=0,
    speaker_device="default",
    sample_rate=16000,
    channels=1,
)

# ❌ Avoid positional arguments
edge = localrtc.Edge("default", 0, "default", 16000, 1)
```

### 2. Device Discovery

```python
# ✅ Enumerate devices before deployment
devices = localrtc.Edge.list_devices()
if devices["audio_inputs"]:
    edge = localrtc.Edge(audio_device=devices["audio_inputs"][0]["index"])
else:
    edge = localrtc.Edge(audio_device="default")

# ❌ Assume device availability
edge = localrtc.Edge(audio_device="USB Microphone")  # May not exist
```

### 3. Audio Format Negotiation

```python
# ✅ Pass agent to enable negotiation
room = await edge.join(agent, room_id="call-1")

# ⚠️ Without agent, uses default format
room = await edge.join(room_id="call-1")  # Defaults to 24kHz mono
```

### 4. Track Publishing

```python
# ✅ Use explicit room and keyword arguments
await edge.publish_tracks(room, audio_track=audio, video_track=video)

# ⚠️ Deprecated legacy convention
await edge.publish_tracks(audio, video)  # Will be removed in v2.0
```

### 5. Resource Cleanup

```python
# ✅ Use context managers
async with agent.join(call):
    await agent.simple_response("Hello!")
    await agent.finish()
# Automatic cleanup

# ✅ Manual cleanup
edge = localrtc.Edge()
try:
    # ... use edge ...
finally:
    await edge.close()

# ❌ Missing cleanup
edge = localrtc.Edge()
# ... use edge ...
# Resources leaked!
```

### 6. Error Handling

```python
# ✅ Handle device errors gracefully
try:
    edge = localrtc.Edge(audio_device="USB Microphone")
except RuntimeError as e:
    logger.warning(f"Failed to open USB Microphone: {e}")
    edge = localrtc.Edge(audio_device="default")

# ✅ Check audio format warnings
# Monitor logs for:
# [AUDIO FORMAT MISMATCH] Input format 44100Hz, 2ch is not directly supported
```

---

## Deprecation Policy

### Current Deprecations

**1. Legacy `publish_tracks()` without room parameter**
- **Deprecated in:** v1.0 (2026-01-22)
- **Removed in:** v2.0
- **Migration:**
  ```python
  # Old
  await edge.publish_tracks(audio, video)

  # New
  await edge.publish_tracks(room, audio_track=audio, video_track=video)
  ```

### Versioning Policy

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **Major version** (X.0.0): Breaking changes to public API
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

### Breaking Change Policy

1. **Deprecation Warning**: Method marked deprecated with `DeprecationWarning`
2. **Documentation**: Migration guide provided in release notes
3. **Grace Period**: Minimum one major version with deprecation warning
4. **Removal**: Deprecated API removed in next major version

**Example Timeline:**
- v1.0: `publish_tracks(audio, video)` deprecated
- v1.x: Deprecation warning issued, both APIs work
- v2.0: Legacy API removed, only new API works

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-22 | Initial stable API release |
|     |            | - Standardized calling conventions |
|     |            | - Comprehensive docstrings |
|     |            | - Required vs optional methods marked |
|     |            | - Deprecation warnings for legacy patterns |

---

## See Also

- [Audio Documentation](./AUDIO_DOCUMENTATION.md) - Comprehensive audio configuration guide
- [Basic Agent Example](../examples/localrtc/basic_agent.py) - Reference implementation
- [Device Discovery Example](../examples/localrtc/device_discovery.py) - Device enumeration patterns
- [Multi-Component Example](../examples/localrtc/multi_component_agent.py) - Advanced composition patterns

---

**Feedback:** For API questions or suggestions, please open an issue at [vision-agents-stream/issues](https://github.com/your-org/vision-agents-stream/issues)
