# Stream + Deepgram STT Example

This example demonstrates how to build a real-time transcription bot that joins a Stream video call and transcribes speech using Deepgram's Speech-to-Text API.

## What it does

- 🤖 Creates a transcription bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🎙️ Transcribes speech in real-time using Deepgram STT
- 📝 Displays transcriptions with timestamps and confidence scores in the terminal

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **Deepgram Account**: Get your API key from [Deepgram Console](https://console.deepgram.com)
3. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/stt_deepgram_transcription
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   Rename `env.example` to `.env` and fill in your actual credentials.

## Usage

Run the example:
```bash
uv run main.py
```

## Partial transcripts

You can receive partial transcript events by setting `interim_results=True` in the `Deepgram` client instantiation.

```python
stt = DeepgramSTT(interim_results=True)
```
