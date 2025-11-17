import pytest
from dotenv import load_dotenv

from vision_agents.plugins import wizper
from conftest import STTSession, skip_blockbuster

# Load environment variables
load_dotenv()


@skip_blockbuster
class TestWizperSTT:
    """Integration tests for Wizper STT"""

    @pytest.fixture
    async def stt(self):
        """Create and manage Wizper STT lifecycle"""
        stt = wizper.STT()
        try:
            yield stt
        finally:
            await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz):
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process the audio
        await stt.process_audio(mia_audio_48khz)

        # Wait for result
        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        # Verify transcript
        full_transcript = session.get_full_transcript()
        assert "forgotten treasures" in full_transcript.lower()
