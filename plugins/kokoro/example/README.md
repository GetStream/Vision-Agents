# Kokoro TTS Example

This directory contains an example demonstrating how to use the Kokoro TTS plugin with Vision Agents.

## Overview

Kokoro is an open-weight, offline TTS model that runs locally without requiring API keys. This makes it ideal for
privacy-focused applications or environments without internet access.

## Features

- **Kokoro TTS**: Open-source, offline text-to-speech that runs locally
- **GetStream**: Real-time communication infrastructure
- **OpenAI LLM**: Intelligent response generation

## Prerequisites

1. **espeak-ng**: Required for phonemization
   ```bash
   # macOS
   brew install espeak-ng

   # Ubuntu/Debian
   sudo apt-get install espeak-ng
   ```

2. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)

3. **OpenAI API Key**: For the LLM component

## Setup

1. Install dependencies:

```bash
cd plugins/kokoro/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
# Required for GetStream
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# Required for OpenAI LLM
OPENAI_API_KEY=your_openai_api_key
```

## Running the Example

```bash
uv run kokoro_example.py run
```

The agent will:

1. Connect to the GetStream edge network
2. Initialize Kokoro TTS (downloads model on first run)
3. Join a call and greet participants when they join

## Customization

### Voice Selection

```python
# Use a specific voice preset
tts = kokoro.TTS(voice="af_heart")

# Use American English
tts = kokoro.TTS(lang_code="a")

# Adjust speech speed
tts = kokoro.TTS(speed=1.2)
```

### Available Voices

See the [Kokoro model card](https://huggingface.co/NeuML/kokoro-int8-onnx#speaker-reference) for available voice
presets.

## Architecture

```
User Joins Call
    ↓
Event Handler (CallSessionParticipantJoinedEvent)
    ↓
LLM generates greeting
    ↓
Kokoro TTS (Local synthesis)
    ↓
User Hears Greeting
```

## Troubleshooting

### No audio output

- Verify espeak-ng is installed: `espeak-ng --version`
- Check that the Kokoro model downloaded successfully
- Ensure GetStream connection is established

### Model download issues

- First run downloads the model (~300MB)
- Ensure you have internet access for the initial download
- Models are cached locally after first download

### Audio quality

- Kokoro outputs at 24kHz sample rate
- Ensure your audio track uses matching sample rate
