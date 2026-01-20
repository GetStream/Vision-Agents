# Fish Audio TTS Examples

This directory contains examples demonstrating how to use the Fish Audio TTS plugin with Vision Agents.

## Examples

### 1. Simple TTS Example (`simple_tts_example.py`)

Basic usage of Fish Audio TTS without a full agent setup. Perfect for testing or simple integrations.

### 2. Full Agent Example (`fish_tts_example.py`)

Complete agent setup with Fish Audio TTS, Deepgram STT, and real-time communication.

## Setup

1. Install dependencies:

```bash
cd plugins/fish/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
# Required for Fish Audio TTS
FISH_AUDIO_API_KEY=your_fish_audio_api_key

# Required for full agent example only:
DEEPGRAM_API_KEY=your_deepgram_api_key
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
```

## Running the Examples

### Full Agent Example

```bash
uv run fish_example.py run
```

## What it does

The example creates an AI agent that:

- Uses **Fish Audio** for high-quality text-to-speech synthesis
- Uses **Deepgram** for speech-to-text transcription
- Uses **GetStream** for real-time communication
- Uses **Smart Turn** detection for natural conversation flow

The agent will greet you using Fish Audio's TTS and be ready to have a conversation.

## Customization

You can customize the Fish Audio TTS settings:

```python
# Use a specific reference voice
tts = fish.TTS(reference_id="your_reference_voice_id")

# Or use a custom endpoint
tts = fish.TTS(base_url="https://your-custom-endpoint.com")
```

## Additional Resources

- [Fish Audio Documentation](https://docs.fish.audio)
- [Vision Agents Documentation](https://visionagents.ai)

