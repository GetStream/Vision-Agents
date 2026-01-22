# Vision Agents Audio Configuration and Troubleshooting Guide

## Overview

This document provides comprehensive guidance on audio format specifications, configuration, and troubleshooting for the Vision Agents framework. Understanding audio format handling is critical for ensuring proper audio quality and compatibility across different transport layers.

## Audio Format Specifications

### Standard Audio Formats

Vision Agents supports the following audio format parameters:

#### Sample Rate (Hz)
- **16000 Hz** - Standard for voice processing and ASR (Automatic Speech Recognition)
- **24000 Hz** - High-quality voice applications
- **48000 Hz** - CD-quality audio, recommended for music and high-fidelity applications
- **Default**: Depends on transport layer (see below)

#### Bit Depth (bits per sample)
- **8-bit** - Low quality, rarely used
- **16-bit** - Standard quality, recommended for most applications
- **24-bit** - High quality
- **32-bit** - Professional quality
- **Default**: 16-bit signed integer

#### Channels
- **1 (Mono)** - Single channel, recommended for voice applications
- **2 (Stereo)** - Dual channel, recommended for music and spatial audio
- **Default**: Depends on transport layer (see below)

### Transport Layer Defaults

Different transport implementations have different default audio configurations:

| Transport Layer | Sample Rate | Channels | Bit Depth | Location |
|----------------|-------------|----------|-----------|----------|
| **GetStream Edge** | 48000 Hz | 2 (Stereo) | 16-bit | `plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py:422-429` |
| **LocalRTC** | 16000 Hz | 1 (Mono) | 16-bit | `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py` |
| **AudioForwarder** | 16000 Hz (fixed) | 1 (Mono) | 16-bit | `agents-core/vision_agents/core/utils/audio_forwarder.py:64` |

### PcmData Type

All audio data in Vision Agents is represented using the `PcmData` dataclass:

```python
from vision_agents.core.types import PcmData

# Example PcmData structure
audio_data = PcmData(
    sample_rate=16000,      # Samples per second (Hz)
    channels=1,             # Number of audio channels
    bit_depth=16,           # Bits per sample
    data=numpy_array,       # Raw PCM audio data as numpy array
    timestamp=1234567890.0  # Unix timestamp (optional)
)
```

**Location**: `agents-core/vision_agents/core/types.py`

## Audio Format Conversion

### Automatic Format Conversion

Vision Agents automatically converts audio formats when needed. The primary conversion implementation is in `AudioOutputTrack._convert_audio()`.

**Location**: `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:537-633`

### Supported Conversions

1. **Sample Rate Conversion** - Uses linear interpolation for resampling
2. **Channel Mixing**:
   - Stereo to Mono: Averages the two channels
   - Mono to Stereo: Duplicates the mono channel
3. **Bit Depth Conversion** - Converts between 8, 16, 24, and 32-bit depths with clipping
4. **Data Type Conversion** - Converts float32 to int16 (GetStream compatibility)

### Format Conversion Example

```python
from vision_agents.plugins.localrtc.tracks import AudioOutputTrack

# Create output track with device defaults
output_track = AudioOutputTrack()

# Audio will be automatically converted to match device requirements
# For example: 48kHz stereo -> 16kHz mono conversion happens transparently
```

## Audio Debug Logging

### Enabling Debug Logging

Set the `VISION_AGENTS_DEBUG_AUDIO` environment variable to enable detailed audio format logging:

```bash
# Enable audio debug logging
export VISION_AGENTS_DEBUG_AUDIO=1

# Run your application
python your_app.py
```

### Debug Output Example

When enabled, you'll see detailed format conversion logs:

```
DEBUG: Audio format conversion:
  Input: 48000 Hz, 2 channels, float32
  Output: 16000 Hz, 1 channel, int16
  Sample rate ratio: 0.333
  Channel mixing: stereo -> mono (averaging)
  Data type conversion: float32 -> int16 with clipping
```

**Location**: Format logging is implemented in `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py:537-633`

### Additional Debugging Tools

```python
# Enable logging in your Python code
import logging
logging.basicConfig(level=logging.DEBUG)

# Create tracks with verbose output
from vision_agents.plugins.localrtc.tracks import AudioInputTrack, AudioOutputTrack

# Input track will log capture format
input_track = AudioInputTrack(sample_rate=16000, channels=1)

# Output track will log playback format and conversions
output_track = AudioOutputTrack()
```

## Common Audio Issues and Troubleshooting

### Issue 1: Sample Rate Mismatch

**Symptoms**: Distorted audio, chipmunk effect, or slow/dragging audio playback

**Cause**: Sample rate mismatch between source and destination

**Solution**:
1. Check the sample rate of your audio source
2. Ensure the transport layer is configured for the correct sample rate
3. Use explicit format conversion if needed

```python
# Example: Configure GetStream edge with specific sample rate
from vision_agents.plugins.getstream import StreamEdge

edge = StreamEdge(connection=conn)
# Create audio track with 16kHz instead of default 48kHz
audio_track = edge.create_audio_track(sample_rate=16000, channels=1)
```

**Reference**: `plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py:422-429`

### Issue 2: Channel Configuration Errors

**Symptoms**: Audio only plays in one speaker, or doubled/echoed audio

**Cause**: Channel count mismatch (mono vs stereo)

**Solution**:
1. Verify source audio channel count
2. Configure transport layer for correct channel count
3. Let AudioOutputTrack handle automatic conversion

```python
# Example: Force mono output
from vision_agents.plugins.localrtc.tracks import AudioOutputTrack

output_track = AudioOutputTrack(channels=1)  # Force mono
```

**Reference**: `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py`

### Issue 3: Audio Clipping/Distortion

**Symptoms**: Crackling, popping, or distorted audio

**Possible Causes**:
1. **Bit depth conversion overflow** - Audio values exceed valid range
2. **Buffer underrun** - Not enough audio data in buffer
3. **Device sample rate mismatch** - Hardware doesn't support requested rate

**Solutions**:

**For bit depth issues**:
- Ensure audio data is normalized to valid range (-1.0 to 1.0 for float32)
- AudioOutputTrack automatically clips values during conversion

**For buffer issues**:
- Increase buffer size (default varies by device)
- Check system CPU usage
- Verify audio data is being generated consistently

**For device compatibility**:
- Query device capabilities before configuration
- Use device defaults when possible

```python
import sounddevice as sd

# List available audio devices and their capabilities
print(sd.query_devices())

# Use device's native format
device_info = sd.query_devices(device_id, 'output')
output_track = AudioOutputTrack(
    sample_rate=int(device_info['default_samplerate']),
    device=device_id
)
```

### Issue 4: No Audio Output

**Symptoms**: Silence, no audio playback

**Diagnostic Steps**:
1. Enable debug logging: `export VISION_AGENTS_DEBUG_AUDIO=1`
2. Check device selection: `python -m sounddevice`
3. Verify audio track is connected to edge transport
4. Confirm audio data is being generated (check PcmData objects)

**Common Fixes**:
```python
# Explicitly specify output device
output_track = AudioOutputTrack(device="MacBook Pro Speakers")

# Verify track is publishing
edge.publish_track(output_track)

# Check if audio subscriber is receiving data
def on_audio(data: PcmData):
    print(f"Received audio: {len(data.data)} samples at {data.sample_rate} Hz")

edge.add_track_subscriber("audio", on_audio)
```

### Issue 5: AudioForwarder Always Outputs 16kHz Mono

**Symptoms**: Audio is always resampled to 16kHz mono regardless of input

**Cause**: AudioForwarder has hardcoded resampling to 16kHz mono for ASR compatibility

**Solution**: This is intentional behavior for voice processing. If you need different formats:
- Use AudioOutputTrack directly instead of AudioForwarder
- Or modify AudioForwarder resampling parameters

**Reference**: `agents-core/vision_agents/core/utils/audio_forwarder.py:64`

### Issue 6: Audio Sync Issues with Video

**Symptoms**: Audio and video are out of sync

**Possible Causes**:
1. Timestamp misalignment
2. Different processing latencies
3. Buffer size differences

**Solutions**:
- Ensure PcmData timestamps are accurate
- Use consistent buffer sizes for audio and video
- Consider implementing AV sync mechanism

## Best Practices

### 1. Use Appropriate Sample Rates

- **Voice applications (ASR, TTS)**: 16kHz mono
- **High-quality voice**: 24kHz mono
- **Music/multimedia**: 48kHz stereo

### 2. Let the Framework Handle Conversions

AudioOutputTrack automatically handles format conversion. Don't manually convert unless necessary:

```python
# Good - let framework handle conversion
output_track = AudioOutputTrack()  # Uses device defaults
edge.publish_track(output_track)

# Avoid - manual conversion adds complexity
# (unless you have specific requirements)
```

### 3. Match Transport Layer Formats

For best performance, match your audio source format to the transport layer defaults:

```python
# For GetStream: 48kHz stereo
edge = StreamEdge(connection=conn)
audio_track = edge.create_audio_track()  # Uses 48kHz stereo default

# For LocalRTC: 16kHz mono
from vision_agents.plugins.localrtc.tracks import AudioInputTrack
input_track = AudioInputTrack()  # Uses 16kHz mono default
```

### 4. Validate Audio Data

Always validate PcmData before processing:

```python
def validate_audio(pcm_data: PcmData) -> bool:
    """Validate PcmData format."""
    if pcm_data.sample_rate not in [8000, 16000, 24000, 48000]:
        raise ValueError(f"Unsupported sample rate: {pcm_data.sample_rate}")

    if pcm_data.channels not in [1, 2]:
        raise ValueError(f"Unsupported channel count: {pcm_data.channels}")

    if pcm_data.bit_depth not in [8, 16, 24, 32]:
        raise ValueError(f"Unsupported bit depth: {pcm_data.bit_depth}")

    return True
```

### 5. Use Debug Logging During Development

Always enable debug logging when developing audio features:

```bash
export VISION_AGENTS_DEBUG_AUDIO=1
```

## Architecture Overview

### Audio Data Flow

```
┌─────────────────┐
│  Audio Source   │ (Microphone, File, TTS)
│  (Various Fmt)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AudioInputTrack │ (Captures/generates audio)
│  or TTS Module  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    PcmData      │ (Standard format container)
│  sample_rate    │
│   channels      │
│   bit_depth     │
│     data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Edge Transport │ (GetStream, LocalRTC)
│  (May convert)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│AudioOutputTrack │ (Format conversion if needed)
│  or Subscriber  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Device or │ (Speaker, File, ASR)
│   Destination   │
└─────────────────┘
```

### Key Components

1. **PcmData** (`agents-core/vision_agents/core/types.py`) - Standard audio data container
2. **EdgeTransport** (`agents-core/vision_agents/core/edge/edge_transport.py`) - Abstract transport interface
3. **StreamEdge** (`plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py`) - GetStream transport implementation
4. **AudioInputTrack/AudioOutputTrack** (`plugins/localrtc/vision_agents/plugins/localrtc/tracks.py`) - Audio capture and playback
5. **AudioForwarder** (`agents-core/vision_agents/core/utils/audio_forwarder.py`) - Audio format adapter for ASR

## Testing Audio Configuration

### Validation Tests

Comprehensive audio format tests are available:

**Location**: `tests/test_audio_format_handling.py`

Run tests:
```bash
pytest tests/test_audio_format_handling.py -v
```

### Manual Testing

Test audio configuration manually:

```python
# Test audio capture
from vision_agents.plugins.localrtc.tracks import AudioInputTrack

input_track = AudioInputTrack(sample_rate=16000, channels=1)
# Capture will start automatically

# Test audio playback
from vision_agents.plugins.localrtc.tracks import AudioOutputTrack
import numpy as np
from vision_agents.core.types import PcmData

output_track = AudioOutputTrack()

# Generate test tone (440 Hz sine wave)
sample_rate = 16000
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)

test_audio = PcmData(
    sample_rate=sample_rate,
    channels=1,
    bit_depth=16,
    data=tone
)

# Play test tone
await output_track.recv()  # AudioOutputTrack will play automatically
```

## Additional Resources

### Related Documentation
- LocalRTC Plugin: `plugins/localrtc/README.md` - Comprehensive audio debugging guide
- GetStream Plugin: `plugins/getstream/README.md` - Transport configuration
- Agent Documentation: `AGENTS.md` - Agent architecture overview

### Source Code Reference
- Core Types: `agents-core/vision_agents/core/types.py`
- Edge Transport: `agents-core/vision_agents/core/edge/edge_transport.py`
- Audio Tracks: `plugins/localrtc/vision_agents/plugins/localrtc/tracks.py`
- Stream Transport: `plugins/getstream/vision_agents/plugins/getstream/stream_edge_transport.py`
- Audio Forwarder: `agents-core/vision_agents/core/utils/audio_forwarder.py`

### Community Support
- Report issues: [GitHub Issues](https://github.com/stream-video/vision-agents/issues)
- Discussions: [GitHub Discussions](https://github.com/stream-video/vision-agents/discussions)

---

**Last Updated**: 2026-01-22
**Version**: 1.0
**Maintained by**: Vision Agents Team
