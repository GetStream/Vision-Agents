# TwelveLabs Pegasus Example

A small, runnable agent that demonstrates the TwelveLabs **Pegasus** `VideoLLM`.

Unlike frame-by-frame VLMs, Pegasus analyzes a short **video clip**, so it can
reason about motion and events over time ("what just happened?"). This example
joins a Stream call, buffers a few seconds of your webcam, and has the agent
describe what it sees out loud.

- `twelvelabs_pegasus_example.py` — Pegasus VLM + Deepgram STT + ElevenLabs TTS,
  joined to a Stream video call.

## Prerequisites

- Python 3.10+
- API keys for:
  - [TwelveLabs](https://twelvelabs.io) — Pegasus video understanding (generous free tier)
  - [Stream](https://getstream.io/try-for-free/) — low-latency video edge
  - [Deepgram](https://deepgram.com/) — speech-to-text
  - [ElevenLabs](https://elevenlabs.io/) — text-to-speech

## Setup

1. From the workspace root, install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file in the workspace root with your keys (see
   [`.env.example`](./.env.example)):
   ```
   TWELVELABS_API_KEY=your_twelvelabs_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   DEEPGRAM_API_KEY=your_deepgram_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   ```

## Running

From the workspace root:

```bash
uv run plugins/twelvelabs/example/twelvelabs_pegasus_example.py
```

Open the join URL printed in the logs in your browser, turn on your camera, and
wait a few seconds — the agent buffers a short clip, sends it to Pegasus, and
speaks a description of what just happened.

## Tuning

Pegasus analyzes a clip rather than a single frame, so a couple of knobs matter
for latency and accuracy:

```python
twelvelabs.PegasusVLM(
    clip_seconds=5,   # length of the analyzed clip (must be >= 4)
    fps=1.0,          # frame sampling rate for the buffer
    max_tokens=512,   # response length (must be >= 512)
)
```

Each request uploads a clip and runs server-side analysis, so latency is higher
than single-frame VLMs. See the [plugin README](../README.md) for the full
parameter list.

## Learn More

- [TwelveLabs Documentation](https://docs.twelvelabs.io/)
- [Vision Agents Documentation](https://visionagents.ai/)
