# Stream + Cartesia Voice Bot Example

This example demonstrates how to build a voice bot that joins a Stream video call, transcribes participants with Cartesia STT, and speaks responses with Cartesia TTS.

## What it does

- Creates a voice bot that joins a Stream video call
- Uses Cartesia Ink for realtime STT and turn detection
- Uses Cartesia Sonic for TTS responses
- Uses OpenAI for the LLM response

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://getstream.io/try-for-free/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents)
2. **Cartesia Account**: Get your API key from [Cartesia](https://cartesia.ai/?utm_medium=partner&utm_source=getstream)
3. **OpenAI Account**: Set an `OPENAI_API_KEY` for the example LLM.
4. **Python 3.10+**: Required for running the example

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

1. **Navigate to this directory:**
   ```bash
   cd plugins/cartesia/example
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   Create a `.env` file with `STREAM_API_KEY`, `STREAM_API_SECRET`, `CARTESIA_API_KEY`, and `OPENAI_API_KEY`.

## Usage

Run the example:

```bash
uv run main.py run
```

Join the generated call, speak into your microphone, and the bot should answer out loud.
