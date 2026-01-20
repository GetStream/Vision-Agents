# ElevenLabs TTS and STT Example

This directory contains an example demonstrating how to use the ElevenLabs TTS and Scribe v2 STT plugins with Vision
Agents.

## Overview

This example creates an AI agent that uses ElevenLabs' state-of-the-art voice technology for both speech synthesis and
recognition.

## Features

- **ElevenLabs TTS**: High-quality, natural-sounding text-to-speech with customizable voices
- **ElevenLabs Scribe v2**: Real-time speech-to-text with low latency (~150ms) and 99 language support
- **GetStream**: Real-time communication infrastructure
- **Smart Turn Detection**: Natural conversation flow management
- **Gemini LLM**: Intelligent response generation

## Setup

1. Install dependencies:

```bash
cd plugins/elevenlabs/example
uv sync
```

2. Create a `.env` file with your API keys:

```bash
# Required for ElevenLabs TTS and STT
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Required for GetStream (real-time communication)
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret

# Required for Gemini LLM
GEMINI_API_KEY=your_gemini_api_key
```

## Running the Example

```bash
uv run elevenlabs_example.py run
```

The agent will:

1. Connect to the GetStream edge network
2. Initialize ElevenLabs TTS and Scribe v2 STT
3. Join a call and greet you
4. Listen and respond to your voice input in real-time

## Customization

### Voice Selection

You can customize the ElevenLabs voice:

```python
# Use a specific voice ID
tts = elevenlabs.TTS(voice_id="your_voice_id")

# Use a different model
tts = elevenlabs.TTS(model_id="eleven_flash_v2_5")
```

### STT Configuration

Customize the speech-to-text settings:

```python
# Use a different language
stt = elevenlabs.STT(language_code="es")  # Spanish

# Adjust VAD settings
stt = elevenlabs.STT(
    vad_threshold=0.5,
    vad_silence_threshold_secs=2.0,
)
```

### Turn Detection

Adjust turn detection sensitivity:

```python
turn_detection = smart_turn.TurnDetection(
    buffer_in_seconds=2.0,  # How long to wait for speech
    confidence_threshold=0.5,  # How confident to be before ending turn
)
```

## ElevenLabs Models

### TTS Models

- `eleven_multilingual_v2`: High-quality, emotionally rich (default)
- `eleven_flash_v2_5`: Ultra-fast with low latency (~75ms)
- `eleven_turbo_v2_5`: Balanced quality and speed

### STT Model

- `scribe_v2_realtime`: Real-time transcription with 99 language support

## Architecture

```
User Voice Input
    ↓
ElevenLabs Scribe v2 STT (Real-time transcription)
    ↓
Gemini LLM (Generate response)
    ↓
ElevenLabs TTS (Synthesize speech)
    ↓
User Hears Response
```

## Additional Resources

- [ElevenLabs Documentation](https://elevenlabs.io/docs)
- [ElevenLabs Voice Library](https://elevenlabs.io/voice-library)
- [Vision Agents Documentation](https://visionagents.ai)
- [GetStream Documentation](https://getstream.io)

## Troubleshooting

### No audio output

- Verify your `ELEVENLABS_API_KEY` is valid
- Check your audio device settings
- Ensure GetStream connection is established

### Poor transcription quality

- Use 16kHz sample rate audio for optimal results
- Speak clearly and avoid background noise
- Adjust `vad_threshold` if needed

### High latency

- Consider using `eleven_flash_v2_5` for TTS
- Check your network connection
- Reduce `buffer_in_seconds` in turn detection

