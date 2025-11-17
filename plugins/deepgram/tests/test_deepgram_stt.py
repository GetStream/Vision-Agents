import pytest

from vision_agents.core.edge.types import Participant
from vision_agents.plugins import deepgram
from conftest import STTSession


class TestDeepgramSTT:
    """Integration tests for Deepgram STT"""

    @pytest.fixture
    async def stt(self):
        """Create and manage Deepgram STT lifecycle"""
        stt = deepgram.STT(eager_turn_detection=True)
        try:
            await stt.start()
            yield stt
        finally:
            await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self, stt, mia_audio_48khz, silence_2s_48khz
    ):
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process the mia audio
        await stt.process_audio(
            mia_audio_48khz, participant=Participant({}, user_id="hi")
        )

        # Send 2 seconds of silence to trigger end of turn
        await stt.process_audio(
            silence_2s_48khz, participant=Participant({}, user_id="hi")
        )

        # Wait for result
        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        # Verify transcript
        full_transcript = session.get_full_transcript()
        assert full_transcript is not None
        assert "forgotten treasures" in full_transcript.lower()

        assert session.transcripts[0].participant.user_id == "hi"
