# Vision Agents - Local RTC Plugin

Local RTC plugin for Vision Agents framework, enabling local audio/video stream processing.

## Installation

```bash
pip install -e plugins/localrtc
```

## Features

- Local audio stream processing
- Low-latency audio I/O
- Compatible with Vision Agents Edge Transport interface

## Dependencies

- sounddevice: For local audio capture and playback
- numpy: For audio data processing

## Configuration

The LocalRTC plugin supports extensive configuration to customize audio/video behavior without code changes. Configuration can be provided in three ways:

### 1. Direct Configuration (Programmatic)

```python
from vision_agents.plugins.localrtc.localedge import LocalEdge, LocalEdgeConfig, AudioConfig, VideoConfig

# Create custom configuration
config = LocalEdgeConfig(
    audio=AudioConfig(
        input_sample_rate=48000,
        output_sample_rate=24000,
        input_channels=2,
        output_channels=1,
        bit_depth=16,
    ),
    video=VideoConfig(
        default_width=1920,
        default_height=1080,
        default_fps=30,
    ),
)

# Initialize LocalEdge with configuration
edge = LocalEdge(config=config)
```

### 2. Environment Variables

All configuration values can be overridden via environment variables with the `VA_` prefix:

```bash
# Audio configuration
export VA_AUDIO_INPUT_SAMPLE_RATE=48000
export VA_AUDIO_OUTPUT_SAMPLE_RATE=24000
export VA_AUDIO_INPUT_CHANNELS=2
export VA_AUDIO_OUTPUT_CHANNELS=1
export VA_AUDIO_BIT_DEPTH=16
export VA_AUDIO_INPUT_BUFFER_DURATION=2.0
export VA_AUDIO_OUTPUT_BUFFER_SIZE_MS=10000
export VA_AUDIO_CAPTURE_CHUNK_DURATION=0.1
export VA_AUDIO_PLAYBACK_CHUNK_DURATION=0.05
export VA_AUDIO_LOOP_SLEEP_INTERVAL=0.01
export VA_AUDIO_FLUSH_POLL_INTERVAL=0.05
export VA_AUDIO_ERROR_RETRY_DELAY=0.1
export VA_AUDIO_THREAD_JOIN_TIMEOUT=2.0
export VA_AUDIO_MAX_INT16_VALUE=32767
export VA_AUDIO_EOS_WAIT_TIME=0.1

# Video configuration
export VA_VIDEO_DEFAULT_WIDTH=1920
export VA_VIDEO_DEFAULT_HEIGHT=1080
export VA_VIDEO_DEFAULT_FPS=30
export VA_VIDEO_FORMAT=BGR
export VA_VIDEO_MAX_BUFFERS=1

# GStreamer configuration
export VA_GSTREAMER_APPSINK_NAME=sink
export VA_GSTREAMER_APPSRC_NAME=src
export VA_GSTREAMER_AUDIO_LAYOUT=interleaved

# Then initialize without explicit config (uses environment variables)
edge = LocalEdge()
```

### 3. Default Configuration

If no configuration is provided, sensible defaults are used:

**Audio Defaults:**
- Input sample rate: 16000 Hz
- Output sample rate: 24000 Hz (Gemini native format)
- Input channels: 1 (mono)
- Output channels: 1 (mono)
- Bit depth: 16-bit
- Input buffer duration: 2.0 seconds
- Output buffer size: 10000 ms (10 seconds)
- Capture chunk duration: 0.1 seconds (100ms)
- Playback chunk duration: 0.05 seconds (50ms)

**Video Defaults:**
- Width: 640 pixels
- Height: 480 pixels
- FPS: 30 frames per second
- Format: BGR (for OpenCV compatibility)

**GStreamer Defaults:**
- Appsink name: "sink"
- Appsrc name: "src"
- Audio layout: "interleaved"

### Configuration Reference

#### AudioConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_sample_rate` | int | 16000 | Audio input sampling rate in Hz |
| `output_sample_rate` | int | 24000 | Default output sample rate (when not negotiated with LLM) |
| `input_channels` | int | 1 | Number of audio input channels |
| `output_channels` | int | 1 | Number of audio output channels |
| `bit_depth` | int | 16 | Audio bit depth in bits (8, 16, 24, or 32) |
| `input_buffer_duration` | float | 2.0 | Input buffer duration in seconds |
| `output_buffer_size_ms` | int | 10000 | Output buffer size in milliseconds |
| `capture_chunk_duration` | float | 0.1 | Audio capture chunk duration in seconds |
| `playback_chunk_duration` | float | 0.05 | Audio playback chunk duration in seconds |
| `loop_sleep_interval` | float | 0.01 | Sleep interval in audio loops (avoid busy-waiting) |
| `flush_poll_interval` | float | 0.05 | Poll interval when flushing audio in seconds |
| `error_retry_delay` | float | 0.1 | Delay before retrying on error in seconds |
| `thread_join_timeout` | float | 2.0 | Thread join timeout in seconds |
| `max_int16_value` | int | 32767 | Maximum value for int16 audio conversion |
| `eos_wait_time` | float | 0.1 | Wait time for GStreamer end-of-stream in seconds |

#### VideoConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_width` | int | 640 | Default video frame width in pixels |
| `default_height` | int | 480 | Default video frame height in pixels |
| `default_fps` | int | 30 | Default frames per second |
| `format` | str | "BGR" | Video format for GStreamer (BGR for OpenCV compatibility) |
| `max_buffers` | int | 1 | Maximum buffers for GStreamer appsink |

#### GStreamerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `appsink_name` | str | "sink" | Name for GStreamer appsink element |
| `appsrc_name` | str | "src" | Name for GStreamer appsrc element |
| `audio_layout` | str | "interleaved" | Audio layout for GStreamer caps |

### Configuration Priority

When multiple configuration sources are used, they are applied in this order (later overrides earlier):

1. Hard-coded defaults in `config.py`
2. Environment variables (if set)
3. Programmatic configuration (if provided to `LocalEdge.__init__()`)

### Use Cases

**High-Quality Audio Recording:**
```python
config = LocalEdgeConfig(
    audio=AudioConfig(
        input_sample_rate=48000,
        bit_depth=24,
        input_buffer_duration=5.0,
    )
)
```

**Low-Latency Video:**
```python
config = LocalEdgeConfig(
    video=VideoConfig(
        default_width=1280,
        default_height=720,
        default_fps=60,
    )
)
```

**Custom GStreamer Pipeline:**
```python
config = LocalEdgeConfig(
    gstreamer=GStreamerConfig(
        appsink_name="custom_sink",
        appsrc_name="custom_source",
    )
)
```

## Audio Format Debugging

The Local RTC plugin includes comprehensive audio format debugging capabilities to help diagnose audio pipeline issues such as sample rate mismatches, channel configuration problems, and format conversion errors.

### Enabling Debug Logging

Set the `VISION_AGENTS_DEBUG_AUDIO` environment variable to enable detailed audio format logging:

```bash
export VISION_AGENTS_DEBUG_AUDIO=true
```

Supported values: `true`, `1`, `yes` (case-insensitive)

### What Gets Logged

When debug logging is enabled, you'll see detailed information at key pipeline stages:

#### 1. Audio Input (Capture)
- Device configuration (device index, sample rate, channels, bit depth)
- Capture parameters (duration, frame count)
- Captured data size and timestamp

#### 2. Audio Ingestion (API Responses)
- Format details (sample rate, channels, bit depth)
- Data size and type (CorePcmData vs StreamPcmData)
- NumPy array shapes and dtypes for StreamPcmData

#### 3. Format Conversion
- When conversion is required vs. not needed
- Source and target formats
- Conversion validation errors (if any)
- Output data size after conversion

#### 4. Output Device Writing
- Device configuration
- Buffer state (size before/after)
- Buffer overflow warnings
- Final playback parameters

#### 5. AudioQueue Operations
- Initial queue configuration
- Format consistency checks
- Sample rate mismatch warnings with remediation hints
- Buffer limit exceeded warnings

### Example Debug Output

```
[AUDIO DEBUG] Capturing audio from input device - device_index=None, duration=1.0s, sample_rate=16000Hz, channels=1, bit_depth=16, frames=16000
[AUDIO DEBUG] Captured audio from input device - data_size=32000 bytes, timestamp=1234567890.123
[AUDIO DEBUG] Ingesting StreamPcmData - sample_rate=24000Hz, channels=1, bit_depth=16, data_size=48000 bytes, samples_shape=(24000,), dtype=float32
[AUDIO DEBUG] Format conversion required - from 24000Hz/1ch to 24000Hz/1ch
[AUDIO DEBUG] No format conversion needed - format matches output track (24000Hz/1ch)
[AUDIO DEBUG] Writing to output device - device_index=None, sample_rate=24000Hz, channels=1, bit_depth=16, data_size=48000 bytes, buffer_size_before=0 bytes
[AUDIO DEBUG] Buffer updated - buffer_size_after=48000 bytes
```

### Format Validation

The plugin automatically validates:

- Sample rates must be positive
- Channel counts must be positive
- Bit depths must be 8, 16, 24, or 32
- Data size must match format specifications
- Buffer overflow conditions

### Error Messages

Enhanced error messages include:

- **Sample Rate Mismatch**: Clear warnings when sample rates don't match, with expected vs. actual values
- **Format Conversion Errors**: Detailed information about source/target formats when conversion fails
- **Device Errors**: Context about which device failed and the audio format being used
- **Buffer Overflow**: Warnings when audio processing can't keep up with real-time requirements

### Troubleshooting Audio Issues

1. **Chipmunk/Slow Audio**: Look for sample rate mismatches in the logs
   - Check `[AUDIO FORMAT MISMATCH]` warnings
   - Verify input and output sample rates match

2. **Choppy/Distorted Audio**: Check buffer overflow warnings
   - Look for `Buffer overflow` messages
   - May indicate CPU limitations or processing bottlenecks

3. **No Audio**: Verify device configuration
   - Check device_index in logs
   - Ensure format conversion completed successfully

4. **Format Errors**: Review validation messages
   - Check for invalid bit depth, sample rate, or channel count
   - Verify data size matches format specifications
