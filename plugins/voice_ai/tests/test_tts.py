import pytest
from dotenv import load_dotenv

from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession
from vision_agents.plugins import voice_ai

load_dotenv()


class TestVoiceAiTTS:
    """Integration tests for Voice.ai TTS."""

    @pytest.fixture
    async def tts(self) -> voice_ai.TTS:
        tts = voice_ai.TTS()
        try:
            yield tts
        finally:
            await tts.close()

    @pytest.mark.integration
    async def test_voice_ai_tts_convert_text_to_audio(self, tts: voice_ai.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Voice.ai."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_voice_ai_tts_convert_text_to_audio_manual_test(
        self, tts: voice_ai.TTS
    ):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
