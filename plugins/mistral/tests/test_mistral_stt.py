import asyncio

import pytest
from dotenv import load_dotenv

from vision_agents.plugins import mistral
from conftest import STTSession

load_dotenv()


class TestMistralSTT:
    """Integration tests for Mistral Voxtral STT."""

    @pytest.mark.integration
    async def test_transcribe_chunked_audio(self, mia_audio_48khz_chunked, participant):
        """Test transcription with chunked audio (simulates real-time streaming)."""
        stt = mistral.STT()
        await stt.start()

        session = STTSession(stt)

        # Send audio in chunks like real-time streaming
        for chunk in mia_audio_48khz_chunked:
            await stt.process_audio(chunk, participant)
            await asyncio.sleep(
                0.001
            )  # Simulate real-time pacing, allow receive task to run

        # Close signals end of audio and triggers final transcript
        await stt.close()

        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert "forgotten treasures" in full_transcript.lower()
