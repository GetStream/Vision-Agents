# Stream Audio Moderation Example

This example shows how to build a real-time **audio moderation bot** that joins a Stream video call, transcribes speech with **Deepgram STT**, and immediately sends each transcript through Stream’s Moderation API.

## What it does

- Spawns a bot that joins any Stream video call
- Streams mic audio to Deepgram for real-time transcription
- Checks every transcript against your Moderation configuration
- Prints the recommended action (flag, allow, block, …) next to each line
- Lets you review previously-flagged content with a one-liner script

## Prerequisites

1. **Stream account** – grab your API key/secret from the [Stream Dashboard](https://dashboard.getstream.io)
2. **Deepgram account** – generate an API key in the [Deepgram Console](https://console.deepgram.com)
3. **Python 3.10+**

> ❕ If you don’t have a Moderation config yet, run `python main.py --setup` once to create a minimal one that flags `INSULT` content.

## Installation

You can use any package manager; the snippet below uses [`uv`](https://docs.astral.sh/uv/):

```bash
cd examples/audio_moderation
uv sync            # install dependencies listed in pyproject.toml
cp env.example .env  # add your credentials
```

## Usage

### 1 · Run the bot
```bash
uv run main.py              # default behaviour – join a call & moderate audio
uv run main.py --setup      # (one-off) create a sample moderation config
```
The script prints a call link in your terminal and opens it in your browser. Join the call, speak, and watch the transcripts + moderation verdicts appear live.

### 2 · Review flagged content later
```bash
uv run view_flagged.py
```
`view_flagged.py` pulls your review queue and dumps each flagged item with timestamp, status, and recommended action.

## Environment variables
See `env.example` for the full list. Minimum required:

```env
STREAM_API_KEY=...
STREAM_API_SECRET=...
DEEPGRAM_API_KEY=...
EXAMPLE_BASE_URL=https://pronto.getstream.io
```

## Cleaning up
The bot automatically deletes the temporary users it creates when the script exits. If you created a demo moderation config with `--setup` and want to remove it later, you can do so from the Stream dashboard.
