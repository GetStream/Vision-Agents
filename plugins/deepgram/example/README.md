# Deepgram TTS Example

This example demonstrates how to use Deepgram's Aura text-to-speech with Vision Agents.

## Setup

1. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:

- `DEEPGRAM_API_KEY` - Your Deepgram API key
- `STREAM_API_KEY` - Your Stream API key
- `STREAM_API_SECRET` - Your Stream API secret
- `GOOGLE_API_KEY` - Your Google API key (for Gemini LLM)

2. Install dependencies:

```bash
uv sync
```

3. Run the example:

```bash
uv run python deepgram_tts_example.py run --call-type audio_room --call-id test
```

## Features

- Uses Deepgram Aura for high-quality text-to-speech
- Uses Deepgram Flux for speech-to-text with turn detection
- Integrates with Stream for real-time communication
- Uses Gemini as the LLM for conversation

## Voice Models

Deepgram offers various Aura voice models. You can customize the voice by passing the `model` parameter:

```python
tts = deepgram.TTS(model="aura-2-thalia-en")  # Default female voice
tts = deepgram.TTS(model="aura-2-orion-en")  # Male voice
```

See [Deepgram TTS Models](https://developers.deepgram.com/docs/tts-models) for all available voices.
