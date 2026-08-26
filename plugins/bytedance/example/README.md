# ByteDance plugin examples

Two examples showing the ByteDance / BytePlus Seed Speech plugin:

- `bytedance_stt_tts_example.py` — a voice agent using ByteDance STT + TTS with a Gemini LLM.
- `bytedance_realtime_example.py` — a Live Interpretation agent that translates speech in real time (subtitles only; drop `mode="s2t"` for translated speech).

## Setup

Set your credentials in a `.env` file:

```bash
BYTEDANCE_API_KEY=your-api-key
STREAM_API_KEY=your-stream-key
STREAM_API_SECRET=your-stream-secret
GOOGLE_API_KEY=your-google-key   # only needed for the STT + TTS example
```

Both examples are preconfigured for a BytePlus regional host
(`voice.ap-southeast-1.bytepluses.com`). If your account is on the mainland
Volcengine console, remove the `HOST` override so the built-in defaults apply.

## Run

```bash
uv run bytedance_stt_tts_example.py
uv run bytedance_realtime_example.py
```
