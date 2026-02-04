# Mistral Voxtral STT Example

This example demonstrates using Mistral Voxtral for real-time speech-to-text transcription with Vision Agents.

## Setup

1. Create a `.env` file with your API keys:

```bash
MISTRAL_API_KEY=your_mistral_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
STREAM_API_KEY=your_stream_api_key
STREAM_API_SECRET=your_stream_api_secret
GOOGLE_API_KEY=your_google_api_key
```

2. Install dependencies:

```bash
uv sync
```

## Running

```bash
uv run python mistral_stt_example.py
```

## Notes

- Mistral Voxtral STT does not have built-in turn detection, so this example uses Smart Turn for turn detection.
- Deepgram is used for TTS since Mistral doesn't offer a TTS service.
