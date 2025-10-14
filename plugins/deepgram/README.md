# Deepgram Speech-to-Text Plugin

A high-quality Speech-to-Text (STT) plugin for Vision agents that uses the Deepgram API.

## Installation

```bash
uv add vision-agents-plugins-deepgram
```

## Usage

```python
from vision_agents.plugins import deepgram
from getstream.video.rtc.track_util import PcmData

# Initialize with API key from environment variable
stt = deepgram.STT()

# Or specify API key directly
stt = deepgram.STT(api_key="your_deepgram_api_key")

# Register event handlers
@stt.on("transcript")
def on_transcript(text, user, metadata):
    print(f"Final transcript from {user}: {text}")

@stt.on("partial_transcript")
def on_partial(text, user, metadata):
    print(f"Partial transcript from {user}: {text}")

# Process audio
pcm_data = PcmData(samples=b"\x00\x00" * 1000, sample_rate=48000, format="s16")
await stt.process_audio(pcm_data)

# When done
await stt.close()
```

## Configuration Options

- `api_key`: Deepgram API key (default: reads from `DEEPGRAM_API_KEY` environment variable)
- `options`: Deepgram options for configuring the transcription.  
See the [Deepgram Listen V1 Connect API documentation](https://github.com/deepgram/deepgram-python-sdk/blob/main/websockets-reference.md#%EF%B8%8F-parameters) for more details.
- `sample_rate`: Sample rate of the audio in Hz (default: 16000)
- `language`: Language code for transcription (default: "en-US")
- `keep_alive_interval`: Interval in seconds to send keep-alive messages (default: 1.0s)
- `connection_timeout`: Timeout to wait for the Deepgram connection to be established before skipping the  in seconds to send keep-alive messages (default: 15.0s)

## Requirements

- Python 3.10+
- deepgram-sdk>=5.0.0,<5.1
- numpy>=2.2.6,<2.3
