import pytest
from dotenv import load_dotenv

from vision_agents.plugins import fast_whisper
from conftest import STTSession

load_dotenv()


class TestFastWhisperSTT:
    """Integration tests for Fast Whisper STT"""

    @pytest.fixture
    async def stt(self):
        """Create and manage Fast Whisper STT lifecycle"""
        stt_instance = fast_whisper.STT(model_size="tiny")
        await stt_instance.warmup()
        try:
            yield stt_instance
        finally:
            await stt_instance.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio(self, stt, mia_audio_16khz):
        """Test transcription with buffering."""
        session = STTSession(stt)

        # Process audio (will be buffered and processed after 1s or 2s interval)
        await stt.process_audio(mia_audio_16khz)

        # Wait for processing to complete
        await session.wait_for_result(timeout=60.0)
        assert not session.errors

        # Verify transcript
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
        assert "forgotten treasures" in full_transcript.lower()
