# Stream + Palabra Voice Bot Example

This example demonstrates how to build a voice bot that joins a Stream video call, transcribes participants with
Deepgram STT, and speaks responses with Palabra AI TTS.

## What it does

- Creates a voice bot that joins a Stream video call
- Uses Deepgram for realtime STT and turn detection
- Uses Palabra for streaming TTS responses
- Uses Gemini for the LLM response

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://getstream.io/try-for-free/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents)
2. **Palabra Account**: Create an API key at [platform.palabra.ai/api-keys](https://platform.palabra.ai/api-keys)
3. **Deepgram Account**: Set a `DEEPGRAM_API_KEY` for STT.
4. **Google AI Account**: Set a `GOOGLE_API_KEY` for the example LLM.
5. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd plugins/palabra/example
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   Copy `.env.example` to `.env` and fill in `STREAM_API_KEY`, `STREAM_API_SECRET`, `PALABRA_API_KEY`,
   `DEEPGRAM_API_KEY`, and `GOOGLE_API_KEY`.

## Usage

Run the voice bot:

```bash
uv run main.py run
```

Join the generated call, speak into your microphone, and the bot should answer out loud.

## Checking TTS on its own

`tts_smoke.py` drives the plugin without a call: it synthesizes a few sentences over one WebSocket session, prints the
time to first audio chunk for each, and writes the result to `palabra_smoke.wav`.

```bash
uv run tts_smoke.py
uv run tts_smoke.py "Anything else you would like to hear"
```

Only `PALABRA_API_KEY` is needed for this one.
