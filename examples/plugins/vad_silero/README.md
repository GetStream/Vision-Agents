# Stream + Silero VAD Example

This example demonstrates how to build a voice activity detection bot that joins a Stream video call and detects when participants are speaking using the Silero VAD model.

## What it does

- 🤖 Creates a VAD bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🎤 Detects when users start and stop speaking
- 📝 Displays speech events with timestamps and duration in the terminal

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/vad_silero
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
