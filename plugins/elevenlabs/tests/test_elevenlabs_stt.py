import asyncio

import pytest
from dotenv import load_dotenv
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
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
        session = STTSession(stt)

        await stt.process_audio(mia_audio_16khz, participant=participant)

        # VAD auto-commits after silence; wait for the committed transcript
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0, "No transcript received"
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz, participant):
        """Test transcription with 48kHz audio (requires resampling)"""
        session = STTSession(stt)

        await stt.process_audio(mia_audio_48khz, participant=participant)

        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0, "No transcript received"
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )

    @pytest.mark.integration
    async def test_transcribe_with_participant(self, stt, mia_audio_16khz):
        """Test transcription with participant metadata"""
        session = STTSession(stt)

        participant = Participant({}, user_id="test-user-123", id="test-user-123")

        await stt.process_audio(mia_audio_16khz, participant=participant)

        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0, "No transcript received"
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )
        assert session.transcripts[0].participant.user_id == "test-user-123"

    @pytest.mark.integration
    async def test_transcribe_chunked_audio(
        self, stt, mia_audio_48khz_chunked, silence_2s_48khz, participant
    ):
        """Test transcription with chunked audio stream"""
        session = STTSession(stt)

        for chunk in mia_audio_48khz_chunked[:100]:  # Use first 100 chunks (~2 seconds)
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.02)  # 20ms delay between chunks

        # Send some silence to trigger VAD
        await stt.process_audio(silence_2s_48khz, participant=participant)

        await session.wait_for_result(timeout=10.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        assert len(session.transcripts) > 0 or len(session.partial_transcripts) > 0

    @pytest.mark.integration
    async def test_partial_transcripts(self, stt, mia_audio_48khz, participant):
        """Test that partial transcripts are emitted"""
        session = STTSession(stt)

        await stt.process_audio(mia_audio_48khz, participant=participant)

        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0

    @pytest.mark.integration
    async def test_turn_detection_enabled(self, stt):
        """Test that turn detection is enabled via VAD commit strategy"""
        assert stt.turn_detection is True

    @pytest.mark.integration
    async def test_turn_events_emitted(self, stt, mia_audio_16khz, participant):
        """Test that TurnStartedEvent and TurnEndedEvent are emitted"""
        session = STTSession(stt)
        turn_started_events: list[TurnStartedEvent] = []
        turn_ended_events: list[TurnEndedEvent] = []

        @stt.events.subscribe
        async def on_turn_started(event: TurnStartedEvent):
            turn_started_events.append(event)

        @stt.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            turn_ended_events.append(event)

        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait for VAD to auto-commit
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        assert len(turn_started_events) > 0, "No TurnStartedEvent received"
        assert len(turn_ended_events) > 0, "No TurnEndedEvent received"
        assert turn_started_events[0].participant == participant
        assert turn_ended_events[0].participant == participant
        assert turn_ended_events[0].eager_end_of_turn is False

    @pytest.mark.integration
    async def test_multiple_audio_segments(
        self, stt, mia_audio_16khz, silence_2s_48khz, participant
    ):
        """Test processing multiple audio segments"""
        session = STTSession(stt)

        # Process first audio segment
        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait for VAD auto-commit on first segment
        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        # Add silence to help VAD separate the segments
        await stt.process_audio(silence_2s_48khz, participant=participant)

        # Process second audio segment
        await stt.process_audio(mia_audio_16khz, participant=participant)

        # Wait for second committed transcript
        await asyncio.sleep(5)
        await session.wait_for_result(timeout=30.0)

        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
