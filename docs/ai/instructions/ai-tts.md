## TTS Plugin Guide

Build a TTS plugin that streams audio and emits events. Keep it minimal and follow the project’s layout conventions.

What to create (PEP 420 structure)

- PEP 420: Do NOT add `__init__.py` in plugin folders. Use this layout:
  - `plugins/<provider>/pyproject.toml` (depends on `vision-agents`)
  - `plugins/<provider>/vision_agents/plugins/<provider>/tts.py`
  - `plugins/<provider>/tests/test_tts.py` (pytest tests at plugin root)
  - `plugins/<provider>/example/` (optional, see `plugins/fish/example/fish_tts_example.py`)

Implementation essentials

- Inherit from `vision_agents.core.tts.tts.TTS`.
- Implement `stream_audio(self, text, ...)` and return a single `PcmData`.

  ```python
  from vision_agents.core.edge.types import PcmData

  async def stream_audio(self, text: str, *_, **__) -> PcmData:
      # If your SDK returns raw bytes for the whole utterance
      audio_bytes = await my_sdk.tts.bytes(text=..., ...)
      return PcmData.from_bytes(audio_bytes, sample_rate=16000, channels=1, format="s16")
  ```

- `stop_audio` can be a no-op (the Agent controls playback):

  ```python
  async def stop_audio(self) -> None:
      logger.info("TTS stop requested (no-op)")
  ```

Sample rate is important

- Pass the provider’s native `sample_rate`, `channels`, and `format` to `PcmData.from_bytes`. The Agent resamples to its output track, but accurate native metadata is required for correct timing and quality.
  - If your SDK is streaming, buffer the audio into a single byte string and return one `PcmData`.

Testing and examples

- Add pytest tests at `plugins/<provider>/tests/test_tts.py`. Keep them simple: assert that `stream_audio` yields `PcmData` and that `send()` emits `TTSAudioEvent`.
- Include a minimal example in `plugins/<provider>/example/` (see `fish_tts_example.py`).

Manual playback check (reusable)

- Use the helper `vision_agents.core.tts.manual_test.manual_tts_to_wav` to generate a WAV and optionally play it with `ffplay`.
- Example inside a plugin test:

  ```python
  import pytest
  from vision_agents.core.tts.manual_test import manual_tts_to_wav
  from vision_agents.plugins import fish

  @pytest.mark.integration
  async def test_manual_tts():
      # Requires FISH_API_KEY or FISH_AUDIO_API_KEY
      tts = fish.TTS()
      path = await manual_tts_to_wav(tts, sample_rate=16000, channels=1)
      print("WAV written to:", path)
  ```

Environment variables

- Provider API keys (plugin-specific). For Fish:
  - `FISH_API_KEY` or `FISH_AUDIO_API_KEY` must be set.
- Optional playback:
  - Set `FFPLAY=1` and ensure `ffplay` is in PATH to auto-play the output WAV.

Test session helper

- To simplify event handling in tests, use `vision_agents.core.tts.testing.TTSSession`:

  ```python
  from vision_agents.core.tts.testing import TTSSession

  tts = MyTTS(...)
  tts.set_output_format(sample_rate=16000, channels=1)
  session = TTSSession(tts)

  await tts.send("Hello")
  result = await session.wait_for_result(timeout=10.0)
  assert not result.errors
  assert result.speeches[0]
  ```


References

- See existing plugins for patterns: `plugins/fish`, `plugins/cartesia`, `plugins/elevenlabs`, `plugins/kokoro`.
