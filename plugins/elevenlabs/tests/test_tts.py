import os
import pytest
import pytest_asyncio

from vision_agents.core.tts.testing import TTSSession
from vision_agents.plugins import elevenlabs
from vision_agents.core.tts.manual_test import manual_tts_to_wav


class TestElevenLabsIntegration:
    @pytest_asyncio.fixture
    async def tts(self) -> elevenlabs.TTS:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            pytest.skip(
                "ELEVENLABS_API_KEY environment variable not set. Add it to your .env file."
            )
        return elevenlabs.TTS(api_key=api_key)

    @pytest.mark.integration
    async def test_elevenlabs_with_real_api(self, tts):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)

        await tts.send("This is a test of the ElevenLabs text-to-speech API.")
        result = await session.wait_for_result(timeout=15.0)

        assert not result.errors
        assert len(result.speeches) > 0

    @pytest.mark.integration
    async def test_elevenlabs_tts_convert_text_to_audio_manual_test(self, tts):
        path = await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
        print("ElevenLabs TTS audio written to:", path)
