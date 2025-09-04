# Stream + VAD + Deepgram STT + OpenAI LLM + ElevenLabs TTS Example

This example demonstrates how to build a real-time AI conversation bot using Stream's Video SDK, Silero VAD (Voice Activity Detection), Deepgram's Speech-to-Text API, OpenAI's GPT models, and ElevenLabs Text-to-Speech API.

## What it does

- 🤖 Creates an AI conversation bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🎤 Uses VAD to detect when users are speaking
- 🎙️ Transcribes speech in real-time using Deepgram STT
- 🧠 Processes transcriptions through OpenAI's GPT model for intelligent responses
- 🔊 Converts AI responses to speech using ElevenLabs TTS
- 📝 Displays conversation flow with timestamps in the terminal

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://dashboard.getstream.io)
2. **Deepgram Account**: Get your API key from [Deepgram Console](https://console.deepgram.com)
3. **ElevenLabs Account**: Get your API key from [ElevenLabs](https://elevenlabs.io)
4. **OpenAI Account**: Get your API key from [OpenAI Platform](https://platform.openai.com)
5. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd examples/llm_audio_conversation
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   Note: This will install the getstream package, the required stream plugins and their dependencies.

3. **Set up environment variables:**

   Rename `examples.env` to `.env` and fill in your actual credentials.

## Usage

Run the example:

   ```bash
   uv run main.py
   ```
