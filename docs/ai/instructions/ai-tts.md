## TTS Plugin Guide

Build a TTS plugin that streams audio and emits events. Keep it minimal and follow the project’s layout conventions.

## What to create

- Make sure to follow PEP 420: Do NOT add `__init__.py` in plugin folders. Use this layout:
  - `plugins/<provider>/pyproject.toml` (depends on `vision-agents`)
  - `plugins/<provider>/vision_agents/plugins/<provider>/tts.py`
  - `plugins/<provider>/tests/test_tts.py` (pytest tests at plugin root)
  - `plugins/<provider>/example/` (optional, see `plugins/fish/example/fish_tts_example.py`)

## Implementation essentials

- Inherit from `vision_agents.core.tts.tts.TTS`.
- Implement `stream_audio(self, text, ...)` and return a single `PcmData`.

  ```python
  from vision_agents.core.edge.types import PcmData

  async def stream_audio(self, text: str, *_, **__) -> PcmData:
      audio_bytes = await my_sdk.tts.bytes(text=..., ...)
      # sample_rate, channels and format depend on what the TTS model returns
      return PcmData.from_bytes(audio_bytes, sample_rate=16000, channels=1, format="s16")
  ```

- `stop_audio` can be a no-op

## __init__

The plugin constructor should:

1. Rely on env vars to fetch credentials
2. export kwargs that allow developers to pass important params to the model itself (eg. model name, voice ID, API URL, ...)
3. if applicable the model or client instance
4. have defaults for all params when possible so that ENV var is enough

## Testing and examples

- Look at `plugins/fish/tests/test_fish_tts.py` as a reference of what tests for a TTS plugins should look like
- Add pytest tests at `plugins/<provider>/tests/test_tts.py`. Keep them simple: assert that `stream_audio` yields `PcmData` and that `send()` emits `TTSAudioEvent`.
- Do not write spec tests with mocks, this is usually not necessary
- Make sure to write at least a couple of integration tests, use `TTSSession` to avoid boiler-plate code in testing
- Verify your implementation does not block the event loop. Import and call `assert_tts_send_non_blocking`:

  ```python
  from vision_agents.core.tts.testing import assert_tts_send_non_blocking

  @pytest.mark.integration
  async def test_provider_non_blocking(tts):
      await assert_tts_send_non_blocking(tts, "Hello from TTS")
  ```
- Include a minimal example in `plugins/<provider>/example/` (see `fish_tts_example.py`).
