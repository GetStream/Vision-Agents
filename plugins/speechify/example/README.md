# Stream + Speechify Voice Bot Example

This example demonstrates how to build a voice bot that joins a Stream video call, transcribes participants with Deepgram STT, and speaks responses with Speechify TTS.

## What it does

- Creates a voice bot that joins a Stream video call
- Uses Deepgram for realtime STT and turn detection
- Uses Speechify for TTS responses
- Uses Gemini for the LLM response

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://getstream.io/try-for-free/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents)
2. **Speechify Account**: Get your API key from [Speechify](https://platform.speechify.ai/api-keys)
3. **Deepgram Account**: Set a `DEEPGRAM_API_KEY` for STT.
4. **Google AI Account**: Set a `GOOGLE_API_KEY` for the example LLM.
5. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd plugins/speechify/example
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   Copy `.env.example` to `.env` and fill in `STREAM_API_KEY`, `STREAM_API_SECRET`, `SPEECHIFY_API_KEY`, `DEEPGRAM_API_KEY`, and `GOOGLE_API_KEY`.

## Usage

Run the example:

```bash
uv run main.py run
```

Join the generated call, speak into your microphone, and the bot should answer out loud.
