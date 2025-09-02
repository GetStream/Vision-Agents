# Stream + Moonshine STT + Silero VAD Example

This example demonstrates how to build a real-time transcription bot that joins a Stream video call, detects speech with Silero VAD, and transcribes it using the Moonshine model.

## What it does

- 🤖 Creates a transcription bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🎤 Uses Silero VAD to detect when users are speaking
- 🎙️ Transcribes speech using the offline Moonshine STT model
- 📝 Displays transcriptions with timestamps in the terminal

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/stt_moonshine_transcription
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

   Note: All dependencies, including the Moonshine ONNX package from GitHub, will be installed automatically with uv sync.


3. **Set up environment variables:**
   Rename `env.example` to `.env` and fill in your Stream credentials.

## Usage

Run the example:
```bash
uv run main.py
```
