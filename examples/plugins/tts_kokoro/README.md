# Stream + Kokoro TTS Bot Example

This example demonstrates how to build a text-to-speech bot that joins a Stream video call and greets participants using the open-source Kokoro model.

## What it does

- 🤖 Creates a TTS bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🔊 Greets users when they join using Kokoro TTS (offline model)
- 🎙️ Sends audio directly to the call in real-time

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **espeak-ng**: Required for phonemization (install via `brew install espeak-ng` on macOS)
3. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/tts_kokoro
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   Rename `env.example` to `.env` and fill in your Stream credentials.

## Usage

Run the example:
```bash
uv run main.py
```
