import pytest
from dotenv import load_dotenv

from vision_agents.plugins import nemotron
from conftest import STTSession

load_dotenv()


class TestNemotronSTT:
    """Integration tests for NVIDIA Nemotron Speech STT"""

    @pytest.fixture
    async def stt(self):
        """Create and manage Nemotron STT lifecycle."""
        stt_instance = nemotron.STT(server_url="http://localhost:8765")
        try:
            await stt_instance.start()
            yield stt_instance
        finally:
            await stt_instance.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio(self, stt, mia_audio_16khz):
        """Test transcription of 16kHz audio."""
        session = STTSession(stt)

        await stt.process_audio(mia_audio_16khz)

        await session.wait_for_result(timeout=10.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
        assert "forgotten treasures" in full_transcript.lower()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz):
        """Test transcription of 48kHz audio (auto-resampled to 16kHz)."""
        session = STTSession(stt)

        await stt.process_audio(mia_audio_48khz)

        await session.wait_for_result(timeout=10.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
        assert "forgotten treasures" in full_transcript.lower()
