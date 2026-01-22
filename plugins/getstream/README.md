# GetStream Plugin

A plugin for Vision Agents that provides GetStream integration for real-time audio/video streaming over WebRTC.

## Installation

```bash
pip install vision-agents-plugins-getstream
```

## Usage

### Basic Setup

```python
from vision_agents.plugins.getstream import StreamEdge
from stream_chat import StreamChat

# Initialize GetStream connection
client = StreamChat(api_key="your_api_key", api_secret="your_api_secret")
connection = client.create_call(...)

# Create edge transport
edge = StreamEdge(connection=connection)
```

### Audio Configuration

The GetStream plugin provides flexible audio configuration through the `create_audio_track()` method.

#### Default Audio Format (Recommended)

```python
# Create audio track with default settings (48kHz stereo, 16-bit)
audio_track = edge.create_audio_track()
```

**Default Format Specifications:**
- **Sample Rate**: 48000 Hz (CD-quality)
- **Channels**: 2 (Stereo)
- **Bit Depth**: 16-bit signed integer

#### Custom Audio Format

```python
# Create audio track with custom format for voice applications
audio_track = edge.create_audio_track(
    sample_rate=16000,  # 16kHz for voice/ASR
    channels=1          # Mono for voice
)
```

**Supported Sample Rates**: 8000, 16000, 24000, 48000 Hz
**Supported Channels**: 1 (mono), 2 (stereo)

#### Publishing Audio

```python
# Publish audio track to the stream
edge.publish_track(audio_track)
```

#### Subscribing to Audio

```python
from vision_agents.core.types import PcmData

def on_audio(audio_data: PcmData):
    """Callback for received audio data."""
    print(f"Received audio: {audio_data.sample_rate} Hz, {audio_data.channels} channels")
    print(f"Audio data shape: {audio_data.data.shape}")
    # Process audio data...

# Subscribe to audio track
edge.add_track_subscriber("audio", on_audio)
```

### Audio Format Conversion

The GetStream plugin automatically converts between different audio formats:

```python
# Example: Receiving 48kHz stereo, converting to 16kHz mono for ASR
from vision_agents.core.utils.audio_forwarder import AudioForwarder

def process_audio(pcm_data: PcmData):
    """Process audio for speech recognition."""
    # AudioForwarder automatically resamples to 16kHz mono
    pass

forwarder = AudioForwarder(process_audio)
edge.add_track_subscriber("audio", forwarder.on_audio_frame)
```

### Complete Example

```python
from vision_agents.plugins.getstream import StreamEdge
from vision_agents.core.types import PcmData
from stream_chat import StreamChat
import asyncio

async def main():
    # Setup GetStream connection
    client = StreamChat(api_key="your_api_key", api_secret="your_api_secret")
    connection = client.create_call(...)

    # Create edge transport
    edge = StreamEdge(connection=connection)

    # Create audio track (48kHz stereo for high-quality streaming)
    audio_track = edge.create_audio_track(
        sample_rate=48000,
        channels=2
    )

    # Publish audio
    edge.publish_track(audio_track)

    # Subscribe to remote audio
    def on_remote_audio(audio_data: PcmData):
        print(f"Received: {len(audio_data.data)} samples")
        # Process or play audio...

    edge.add_track_subscriber("audio", on_remote_audio)

    # Run event loop
    await edge.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Audio Troubleshooting

### Enable Debug Logging

```bash
export VISION_AGENTS_DEBUG_AUDIO=1
python your_app.py
```

### Common Issues

**Issue**: Audio sounds distorted or has wrong pitch
- **Cause**: Sample rate mismatch
- **Solution**: Ensure sample rate matches your audio source, or let the framework handle conversion

**Issue**: Audio only plays in one channel
- **Cause**: Channel count mismatch (mono vs stereo)
- **Solution**: Configure channels parameter explicitly

**Issue**: No audio received
- **Cause**: Track not subscribed or published
- **Solution**: Verify `publish_track()` and `add_track_subscriber()` are called

For more detailed troubleshooting, see the main [Audio Documentation](../../AUDIO_DOCUMENTATION.md).

## API Reference

### StreamEdge.create_audio_track()

Creates an audio track for publishing to the stream.

**Parameters:**
- `sample_rate` (int, optional): Audio sample rate in Hz. Default: 48000
- `channels` (int, optional): Number of audio channels. Default: 2

**Returns:** `AudioTrack` instance

**Example:**
```python
audio_track = edge.create_audio_track(sample_rate=16000, channels=1)
```

### StreamEdge.publish_track()

Publishes an audio or video track to the stream.

**Parameters:**
- `track`: Audio or video track to publish

**Example:**
```python
edge.publish_track(audio_track)
```

### StreamEdge.add_track_subscriber()

Subscribes to audio or video tracks from the stream.

**Parameters:**
- `track_type` (str): Type of track ("audio" or "video")
- `callback` (callable): Function called when data is received, receives `PcmData` for audio

**Example:**
```python
def on_audio(data: PcmData):
    print(f"Received {len(data.data)} audio samples")

edge.add_track_subscriber("audio", on_audio)
```

## Development

This plugin follows the standard Vision Agents plugin structure.

### Running Tests

```bash
pytest plugins/getstream/tests/ -v
```

## Additional Resources

- [Audio Configuration Guide](../../AUDIO_DOCUMENTATION.md) - Comprehensive audio format documentation
- [GetStream Documentation](https://getstream.io/video/docs/) - Official GetStream API docs
- [Vision Agents Documentation](../../README.md) - Main framework documentation
