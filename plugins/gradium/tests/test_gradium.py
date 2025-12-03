import pytest
import pytest_asyncio
from dotenv import load_dotenv

from vision_agents.plugins import gradium
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession
from conftest import STTSession


load_dotenv()


class TestGradiumPlugin:
    def test_regular(self):
        assert True

    @pytest.mark.integration
    async def test_simple(self):
        assert True


class TestGradiumSTT:
    """Integration tests for Gradium STT."""

    @pytest.fixture
    async def stt(self):
        """Create and manage Gradium STT lifecycle."""
        stt = gradium.STT(base_url="https://us.api.gradium.ai/api/")
        await stt.start()
        try:
            yield stt
        finally:
            await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio(self, stt, mia_audio_16khz):
        """Test transcription with 16kHz audio."""
        session = STTSession(stt)

        await stt.process_audio(mia_audio_16khz)
        # Signal end of audio to allow processing to complete
        await stt.end_audio()

        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert "forgotten treasures" in full_transcript.lower()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz):
        """Test transcription with 48kHz audio (will be resampled to 24kHz)."""
        session = STTSession(stt)

        await stt.process_audio(mia_audio_48khz)
        # Signal end of audio to allow processing to complete
        await stt.end_audio()

        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert "forgotten treasures" in full_transcript.lower()


class TestGradiumTTS:
    """Integration tests for Gradium TTS."""

    @pytest_asyncio.fixture
    async def tts(self) -> gradium.TTS:
        """Create Gradium TTS instance."""
        return gradium.TTS()

    @pytest.mark.integration
    async def test_gradium_tts_convert_text_to_audio_manual_test(self, tts: gradium.TTS):
        """Manual test that saves audio to WAV file for inspection."""
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)

    @pytest.mark.integration
    async def test_gradium_tts_convert_text_to_audio(self, tts: gradium.TTS):
        """Test basic text-to-speech conversion."""
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Gradium TTS."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_gradium_tts_with_speed_control(self, tts: gradium.TTS):
        """Test TTS with speed control."""
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Testing speed control with Gradium."

        # Test faster speech
        await tts.send(text, speed=-2.0)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0
