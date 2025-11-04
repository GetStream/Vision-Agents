import pytest
import pytest_asyncio

from vision_agents.core.tts.manual_test import manual_tts_to_wav


class TestKokoroIntegration:
    @pytest_asyncio.fixture
    async def tts(self):  # returns kokoro TTS if available
        try:
            import kokoro  # noqa: F401
        except Exception:
            pytest.skip("kokoro package not installed; skipping manual playback test.")
        from vision_agents.plugins import kokoro as kokoro_plugin

        return kokoro_plugin.TTS()

    @pytest.mark.integration
    async def test_kokoro_tts_convert_text_to_audio_manual_test(self, tts):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
