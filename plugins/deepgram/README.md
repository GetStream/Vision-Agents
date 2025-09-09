# Deepgram Speech-to-Text Plugin

A high-quality Speech-to-Text (STT) plugin for GetStream that uses the Deepgram API.

## Installation

```bash
pip install getstream-plugins-deepgram
```

## Usage

```python
from getstream.plugins.deepgram import DeepgramSTT

# Initialize with API key from environment variable
stt = DeepgramSTT()

# Or specify API key directly
stt = DeepgramSTT(api_key="your_deepgram_api_key")

# Register event handlers
@stt.on("transcript")
def on_transcript(text, user, metadata):
    print(f"Final transcript from {user}: {text}")

@stt.on("partial_transcript")
def on_partial(text, user, metadata):
    print(f"Partial transcript from {user}: {text}")

# Process audio
await stt.process_audio(pcm_data)

# When done
await stt.close()
```

## Configuration Options

- `api_key`: Deepgram API key (default: reads from DEEPGRAM_API_KEY environment variable)
- `options`: Deepgram LiveOptions for configuring the transcription
- `sample_rate`: Sample rate of the audio in Hz (default: 16000)
- `language`: Language code for transcription (default: "en-US")
- `keep_alive_interval`: Interval in seconds to send keep-alive messages (default: 5.0)

## Requirements

- Python 3.10+
- deepgram-sdk>=4.5.0
- numpy>=2.2.6,<2.3
