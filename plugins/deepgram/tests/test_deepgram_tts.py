import pytest
from dotenv import load_dotenv

from vision_agents.plugins import deepgram
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession

load_dotenv()


class TestDeepgramTTS:
    """Integration tests for Deepgram TTS."""

    @pytest.fixture
    async def tts(self) -> deepgram.TTS:
        return deepgram.TTS()

    @pytest.mark.integration
    async def test_deepgram_tts_convert_text_to_audio(self, tts: deepgram.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Deepgram."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_deepgram_tts_convert_text_to_audio_manual_test(
        self, tts: deepgram.TTS
    ):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
