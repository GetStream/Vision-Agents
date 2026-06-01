import pytest


@pytest.mark.integration
class TestKokoroIntegration:
    @pytest.fixture
    async def tts(self):  # returns kokoro TTS if available
        try:
            import kokoro  # noqa: F401
        except Exception:
            pytest.skip("kokoro package not installed; skipping manual playback test.")
        from vision_agents.plugins import kokoro as kokoro_plugin

        return kokoro_plugin.TTS()

    async def test_kokoro_tts_convert_text_to_audio(self, tts):
        text = "Hello from Kokoro TTS."

        out = []
        async for item in tts.send_iter(text):
            out.append(item)

        assert len(out) > 0
