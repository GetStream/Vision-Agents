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
