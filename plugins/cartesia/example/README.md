# Stream + Cartesia TTS Bot Example

This example demonstrates how to build a text-to-speech bot that joins a Stream video call and greets participants
using [Cartesia's](https://cartesia.ai/?utm_medium=partner&utm_source=getstream) Sonic voices.

## What it does

- 🤖 Creates a TTS bot that joins a Stream video call
- 🌐 Opens a browser interface for users to join the call
- 🔊 Greets users when they join using Cartesia TTS
- 🎙️ Sends audio directly to the call in real-time

## Prerequisites

1. **Stream Account**: Get your API credentials from [Stream Dashboard](https://getstream.io/try-for-free/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents)
2. **Cartesia Account**: Get your API key from [Cartesia](https://cartesia.ai/?utm_medium=partner&utm_source=getstream)
3. **Python 3.10+**: Required for running the example

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
   Rename `env.example` to `.env` and fill in your actual credentials.

## Usage

Run the example:

```bash
uv run main.py run
```
