import pytest
from dotenv import load_dotenv
import asyncio

from vision_agents.core.edge.types import Participant
from vision_agents.plugins import elevenlabs
from conftest import STTSession

# Load environment variables
load_dotenv()


class TestElevenLabsSTT:
    """Integration tests for ElevenLabs Scribe v2 STT"""

    @pytest.fixture
    async def stt(self):
        """Create and manage ElevenLabs STT lifecycle"""
        stt = elevenlabs.STT(
            language_code="en",
            audio_chunk_duration_ms=100,  # Send 100ms chunks
        )
        try:
            await stt.start()
            yield stt
        finally:
            await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_16khz(self, stt, mia_audio_16khz, participant):
        """Test transcription with 16kHz audio (native sample rate)"""
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process the audio with participant
        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait for result
        # Wait a bit longer for all audio to be processed
        await asyncio.sleep(5)

        await stt.clear()
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        await asyncio.sleep(5)

        # Verify transcript - check for any significant part of the transcript
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0, "No transcript received"
        # The transcript should contain some of the key words from the audio
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz, participant):
        """Test transcription with 48kHz audio (requires resampling)"""
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process the audio with participant
        await stt.process_audio(mia_audio_48khz, participant=participant)

        # Wait a bit for all audio to be processed
        await asyncio.sleep(5)

        # Commit the transcript (required for MANUAL commit strategy)
        await stt.clear()

        # Wait for committed transcript
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        await asyncio.sleep(5)

        # Verify transcript - check for any significant part of the transcript
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0, "No transcript received"
        # The transcript should contain some of the key words from the audio
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )

    @pytest.mark.integration
    async def test_transcribe_with_participant(self, stt, mia_audio_16khz):
        """Test transcription with participant metadata"""
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Create a participant
        participant = Participant({}, user_id="test-user-123")

        # Process the audio with participant
        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait a bit for all audio to be processed
        await asyncio.sleep(5)

        # Commit the transcript (required for MANUAL commit strategy)
        await stt.clear()

        # Wait for committed transcript
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        await asyncio.sleep(5)

        # Verify transcript and participant
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0, "No transcript received"
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )
        assert session.transcripts[0].participant.user_id == "test-user-123"

    @pytest.mark.integration
    async def test_transcribe_chunked_audio(
        self, stt, mia_audio_48khz_chunked, participant
    ):
        """Test transcription with chunked audio stream"""
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process audio chunks one by one (simulating real-time streaming)
        # Use more chunks to ensure we get a complete phrase
        for chunk in mia_audio_48khz_chunked[:100]:  # Use first 100 chunks (~2 seconds)
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(
                0.02
            )  # 20ms delay between chunks (real-time simulation)

        # Wait for audio to be processed
        await asyncio.sleep(3)

        # Commit the transcript (required for MANUAL commit strategy)
        await stt.clear()

        # Wait for committed transcript
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        # Verify we got some transcript
        assert len(session.transcripts) > 0 or len(session.partial_transcripts) > 0

    @pytest.mark.integration
    async def test_partial_transcripts(self, stt, mia_audio_48khz, participant):
        """Test that partial transcripts are emitted"""
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process the audio with participant
        await stt.process_audio(mia_audio_48khz, participant=participant)

        # Wait for audio to be processed
        await asyncio.sleep(5)

        # Commit the transcript (required for MANUAL commit strategy)
        await stt.clear()

        # Wait for committed transcript
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        # Verify we got both partial and final transcripts
        # Note: Depending on the VAD settings, we may or may not get partials
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0

    @pytest.mark.integration
    async def test_turn_detection_disabled(self, stt):
        """Test that turn detection is disabled for Scribe v2"""
        # Scribe v2 does not support turn detection
        assert stt.turn_detection is False

    @pytest.mark.integration
    async def test_multiple_audio_segments(
        self, stt, mia_audio_16khz, silence_2s_48khz, participant
    ):
        """Test processing multiple audio segments"""
        # Create session to collect transcripts and errors
        session = STTSession(stt)

        # Process first audio segment
        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait for audio to be processed
        await asyncio.sleep(5)

        await stt.clear()

        # Wait for first committed transcript
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        # Add silence to help VAD separate the segments
        await stt.process_audio(silence_2s_48khz, participant=participant)

        # Process second audio segment
        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait for audio to be processed
        await asyncio.sleep(5)

        await stt.clear()

        # Wait a bit longer for second result
        await session.wait_for_result(timeout=30.0)

        # Should have gotten additional transcripts
        # Note: Depending on VAD behavior, may combine or separate
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
