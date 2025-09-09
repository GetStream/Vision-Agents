# Cartesia Text-to-Speech Plugin

High-quality **Text-to-Speech** (TTS) plugin for [GetStream](https://getstream.io/) backed by the
[Cartesia](https://github.com/cartesia-ai/cartesia-python) Sonic model. It lets a Python
process speak PCM audio into a Stream call.

## Installation

Install from PyPI (installs both `getstream` and the Cartesia SDK):

```bash
pip install "getstream-plugins-cartesia[webrtc]"
```

If you already have the Stream SDK in your project just add the Cartesia plugin:

```bash
pip install cartesia getstream-plugins-cartesia
```

## Usage

```python
from getstream.plugins.cartesia import CartesiaTTS
from getstream.video.rtc.audio_track import AudioStreamTrack

async def speak():
    # Option A: read key from env var (CARTESIA_API_KEY)
    tts = CartesiaTTS()

    # Option B: pass explicitly
    # tts = CartesiaTTS(api_key="<your key>")

    # Audio MUST be 16-kHz, 16-bit PCM (matches Cartesia Sonic model)
    track = AudioStreamTrack(framerate=16000)
    tts.set_output_track(track)

    # Listen for every raw PCM chunk that gets produced
    @tts.on("audio")
    def _on_audio(chunk: bytes, user):
        print("🔊 got", len(chunk), "bytes of audio")

    await tts.send("Hello from Cartesia!")

# Run inside an event-loop, e.g. `asyncio.run(speak())`
```

## Configuration Options

- `api_key` (str, optional) – Cartesia API key (falls back to `CARTESIA_API_KEY`).
- `model_id` (str) – Which model to hit (`"sonic-2"` by default).
- `voice_id` (str | None) – Cartesia voice to use (pass `None` for model default).
- `sample_rate` (int) – Target sample-rate in Hz. Must match the
  `AudioStreamTrack.framerate` you attach (defaults to `16000`). If they don't match a
  `TypeError` is raised early so you don't get distorted audio on the call.

Events emitted:

• `audio` – each raw PCM chunk, arguments: `chunk: bytes`, `user: dict | None`

• `error` – any exception raised during synthesis

## Requirements

- Python 3.10+
- `cartesia>=2.0.5` (automatically installed)

## Testing

Run the offline unit-tests:

```bash
pytest -q getstream/plugins/cartesia/tests
```

To additionally exercise the live Cartesia API set `CARTESIA_API_KEY` in your
environment; the integration test will be executed automatically.

---

💡 See `examples/tts_cartesia/` for a fully-working bot that joins a Stream call
and greets participants using this plugin.
