# Stream + FastMCP Example

This example demonstrates how to build an AI conversation bot with tool-calling capabilities using Stream's Video SDK, FastMCP, and multiple AI services (STT, LLM, TTS).

## What it does

- 🤖 Creates an AI conversation bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🎤 Uses Deepgram STT to transcribe speech in real-time
- 🧠 Processes transcriptions through OpenAI's GPT model with MCP tool access
- 🛠️ Supports function calls (example: weather forecasts)
- 🔊 Converts AI responses to speech using ElevenLabs TTS
- 📝 Displays conversation flow with timestamps in the terminal

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **OpenAI Account**: Get your API key from [OpenAI Platform](https://platform.openai.com)
3. **Deepgram Account**: Get your API key from [Deepgram Console](https://console.deepgram.com)
4. **ElevenLabs Account**: Get your API key from [ElevenLabs](https://elevenlabs.io)
5. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/mcp
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

The bot will automatically start the MCP server, join the call, and begin listening for speech. Try asking for a weather forecast to see the tool-calling functionality in action.
