# Silero Voice Activity Detection Plugin

A fast and accurate Voice Activity Detection (VAD) plugin for GetStream that uses the Silero VAD model.

## Installation

```bash
pip install getstream-plugins-silero
```

## Usage

```python
from getstream.plugins.silero import SileroVAD
from getstream.video.rtc.track_util import PcmData

# Initialize with default settings
vad = SileroVAD()

# Or customize parameters
vad = SileroVAD(
    sample_rate=16000,
    frame_size=512,
    silence_threshold=0.3,
    speech_pad_ms=300,
    min_speech_ms=250,
    max_speech_ms=60000,
)

# Register event handlers
@vad.on("audio")
async def on_audio(pcm_data, user):
    print(f"Detected speech: {pcm_data.duration:.2f} seconds")
    # Process the detected speech with an STT engine
    # await stt.process_audio(pcm_data)

# Process incoming audio
incoming_audio = PcmData(samples=audio_bytes, sample_rate=16000, format="s16")
await vad.process_audio(incoming_audio)

# Reset state if needed
await vad.reset()
```

## Configuration Options

- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `frame_size`: Size of audio frames to process (default: 512)
- `silence_threshold`: Threshold for detecting silence (0.0 to 1.0) (default: 0.5)
- `speech_pad_ms`: Number of milliseconds to pad before/after speech (default: 300)
- `min_speech_ms`: Minimum milliseconds of speech to emit (default: 250)
- `max_speech_ms`: Maximum milliseconds of speech before forced flush (default: 30000)
