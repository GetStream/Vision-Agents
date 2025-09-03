# Stream + OpenAI Realtime Speech-to-Speech Example

This example demonstrates how to connect an OpenAI Realtime speech-to-speech agent to a Stream video call for real-time voice conversations with function calling capabilities.

## What it does

- 🤖 Creates an AI agent that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🎤 Listens to speech and responds with synthesized voice in real-time
- 🛠️ Supports function calls (example: "Start closed captions")
- 🧠 Uses OpenAI's GPT-4o Realtime model with voice capabilities

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **OpenAI Account**: Get your API key from [OpenAI Platform](https://platform.openai.com) (must have Realtime access)
3. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/openai_realtime_speech_to_speech
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
