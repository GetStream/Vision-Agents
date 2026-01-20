# Pocket TTS Example

A Vision Agents example using Pocket TTS for local text-to-speech.

## Features

- Pocket TTS runs locally on CPU (no TTS API key needed)
- ~200ms latency, 10x+ faster than real-time
- Voice cloning support

## Requirements

Set the following environment variables:

- `DEEPGRAM_API_KEY` - for speech-to-text
- `STREAM_API_KEY` and `STREAM_API_SECRET` - for real-time communication
- `GOOGLE_API_KEY` - for Gemini LLM

## Running the Example

```bash
cd plugins/pocket/example
uv run pocket_example.py run
```
