We use pytest for testing. Be sure to mark integration tests with @pytest.mark.integration
Async is automatic, no need to tag that.
Keep tests short and don't use mocking unless explicitly asked to use mocks.

This project uses uv to manage Python and its dependencies so when you run tests, make sure to use uv run and not python -m

Extend from BaseTest

Store data for fixtures in tests/test_assets/...

Non-blocking checks

- TTS plugins must not block the event loop inside `stream_audio`. Use the helper in `vision_agents.core.tts.testing`:

  ```python
  from vision_agents.core.tts.testing import assert_tts_send_non_blocking

  @pytest.mark.integration
  async def test_tts_non_blocking(tts):
      await assert_tts_send_non_blocking(tts, "Hello")
  ```
