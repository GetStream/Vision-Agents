# Fish Audio Plugin

A high-quality Text-to-Speech (TTS) and Speech-to-Text (STT) plugin for Vision Agents that uses the Fish Audio API.

## Installation

```bash
pip install vision-agents-plugins-fish
```

## Usage

### Text-to-Speech (TTS)

```python
from vision_agents.plugins.fish import TTS
from getstream.video.rtc.audio_track import AudioStreamTrack

# Initialize with API key from environment variable
tts = TTS()

# Or specify API key directly
tts = TTS(api_key="your_fish_audio_api_key")

# Create an audio track to output speech
track = AudioStreamTrack(framerate=16000)
tts.set_output_track(track)

# Register event handlers
@tts.events.subscribe
async def on_audio(event):
    print(f"Received audio chunk: {len(event.audio_data)} bytes")

# Send text to be converted to speech
await tts.send("Hello, this is a test of the Fish Audio text-to-speech plugin.")
```

### Speech-to-Text (STT)

```python
from vision_agents.plugins.fish import STT
from getstream.video.rtc.track_util import PcmData

# Initialize with API key from environment variable
stt = STT()

# Or specify API key directly and language
stt = STT(api_key="your_fish_audio_api_key", language="en")

# Register event handlers
@stt.events.subscribe
async def on_transcript(event):
    print(f"Transcript: {event.text}")

# Process audio data
pcm_data = PcmData(samples=audio_samples, sample_rate=16000)
await stt.process_audio(pcm_data)
```

## Configuration Options

### TTS Options

- `api_key`: Fish Audio API key (default: reads from `FISH_API_KEY` environment variable)
- `reference_id`: Optional reference voice ID to use for synthesis
- `base_url`: Optional custom API endpoint (default: uses Fish Audio's default endpoint)
- `client`: Optionally pass in your own instance of the Fish Audio Session

### STT Options

- `api_key`: Fish Audio API key (default: reads from `FISH_API_KEY` environment variable)
- `language`: Language code for transcription (e.g., "en", "zh"). If None, automatic language detection will be used
- `ignore_timestamps`: Skip timestamp processing for faster results (default: False)
- `sample_rate`: Sample rate of the audio in Hz (default: 16000)
- `base_url`: Optional custom API endpoint
- `client`: Optionally pass in your own instance of the Fish Audio Session

## Reference Audio

Fish Audio supports using reference audio for voice cloning:

```python
from vision_agents.plugins.fish import TTS

# Using a reference voice ID
tts = TTS(reference_id="your_reference_voice_id")

# Or pass reference audio dynamically when sending text
# (See Fish Audio SDK documentation for advanced usage)
```

## Supported Languages (STT)

Fish Audio STT supports multiple languages with automatic detection. Common language codes include:
- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `pt` - Portuguese

For automatic language detection, set `language=None` (default).

## Supported Audio Formats (STT)

The STT implementation accepts PCM audio data and converts it to WAV format internally. Supported configurations:
- Maximum audio size: 100MB
- Maximum duration: 60 minutes
- Sample rate: 16kHz or higher recommended
- Format: Mono, 16-bit PCM

## Requirements

- Python 3.10+
- fish-audio-sdk>=2025.4.2

## Getting Your API Key

1. Sign up for a Fish Audio account at [https://fish.audio](https://fish.audio)
2. Navigate to the API Keys section in your dashboard
3. Create a new API key
4. Set the `FISH_API_KEY` environment variable or pass it directly to the plugin

