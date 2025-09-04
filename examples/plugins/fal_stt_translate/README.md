# Stream + FAL.ai Wizper STT Example

This sample shows how to build a real-time transcription and translation bot that joins a
Stream video call and converts speech to text (or translates it) using
[FAL.ai](https://fal.ai) — specifically the **Wizper** Speech-to-Text model.

## What it does

- Creates a **transcription bot** that automatically joins a Stream video call
- Opens a **browser link** so participants can join the call in one click
- Uses **Silero VAD** to detect speech segments and sends them to FAL.ai
- Prints final transcripts (optionally translated) in the terminal with
  timestamps and confidence values

## Prerequisites

1. **Stream account** – Get your API key & secret from the
   [Stream Dashboard](https://dashboard.getstream.io)
2. **FAL.ai account** – Create an API key from the
   [FAL Console](https://app.fal.ai)
3. **Python 3.10+**

## Installation

You can use your preferred package manager, but we recommend [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Navigate to this directory
cd examples/stt_fal_transcription

# 2. Install dependencies
uv sync
```

## Environment variables

Copy `env.example` to `.env` and fill in your credentials:

```
STREAM_API_KEY=<your stream key>
STREAM_API_SECRET=<your stream secret>
FAL_KEY=<your fal api key>

# Optional – where the browser should open the call link
EXAMPLE_BASE_URL=https://getstream.io/video/examples
```

## Usage

Run the example:

```bash
uv run main.py
```

What happens next:

1. A unique call ID is generated.
2. Your default browser opens so you can join the call as a regular user.
3. The bot user joins silently and begins listening for speech.
4. Completed speech turns are transcribed (or translated) and printed.

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop the bot.

## Customising the STT behaviour

The STT setup is in `main.py`:

```python
stt = FalWizperSTT(task="transcribe", target_language="fr")
```

Key parameters:

- `task`: `"transcribe"` (default) or `"translate"`
- `target_language`: ISO-639-1 code used when `task="translate"`

Example – translate everything to Spanish:

```python
stt = FalWizperSTT(task="translate", target_language="es")
```

Partial transcripts are not currently supported by Wizper; callbacks fire when
FAL.ai marks a segment as complete.

## Troubleshooting

- Double-check that your Stream and FAL keys are correct and present in `.env`.
- Ensure no other process is using the microphone if you run this on a local
  machine.
