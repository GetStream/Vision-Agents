import asyncio

import pytest
from dotenv import load_dotenv

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins import elevenlabs

load_dotenv()


class TestElevenLabsSTT:
    """Integration tests for ElevenLabs Scribe v2 STT"""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    async def stt(self):
        stt = elevenlabs.STT(
            language_code="en",
            audio_chunk_duration_ms=100,
        )
        await stt.start()
        yield stt
        await stt.close()

    @pytest.fixture
    async def stt_short_keepalive(self):
        stt = elevenlabs.STT(
            language_code="en",
            audio_chunk_duration_ms=100,
            keepalive_interval_ms=500,
        )
        await stt.start()
        yield stt
        await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_16khz(self, stt, mia_audio_16khz, participant):
        """Test transcription with 16kHz audio (native sample rate)"""
        await stt.process_audio(mia_audio_16khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        transcripts = [i for i in items if isinstance(i, Transcript)]
        full_transcript = " ".join(t.text for t in transcripts if t.final)
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz, participant):
        """Test transcription with 48kHz audio (requires resampling)"""
        await stt.process_audio(mia_audio_48khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        transcripts = [i for i in items if isinstance(i, Transcript)]
        full_transcript = " ".join(t.text for t in transcripts if t.final)
        assert any(
            word in full_transcript.lower()
            for word in ["village", "quiet", "mia", "treasures"]
        )
        assert transcripts[0].participant == participant

    @pytest.mark.integration
    async def test_transcribe_chunked_audio(
        self, stt, mia_audio_48khz_chunked, silence_2s_48khz, participant
    ):
        """Test transcription with chunked audio stream"""
        for chunk in mia_audio_48khz_chunked[:100]:
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.02)

        await stt.process_audio(silence_2s_48khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert transcripts

    @pytest.mark.integration
    async def test_turn_detection_enabled(self, stt):
        assert stt.turn_detection is True

    @pytest.mark.integration
    async def test_turn_events_emitted(self, stt, mia_audio_16khz, participant):
        """One TurnStarted and exactly one TurnEnded per utterance."""
        await stt.process_audio(mia_audio_16khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        turn_started = [i for i in items if isinstance(i, TurnStarted)]
        turn_ended = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turn_started) == 1
        assert len(turn_ended) == 1

    @pytest.mark.integration
    async def test_multiple_audio_segments(
        self, stt, mia_audio_16khz, silence_2s_48khz, participant
    ):
        """Test processing multiple audio segments"""
        await stt.process_audio(mia_audio_16khz, participant=participant)
        await stt.process_audio(silence_2s_48khz, participant=participant)
        await stt.process_audio(mia_audio_16khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        full_transcript = " ".join(t.text for t in finals)
        assert len(full_transcript) > 0

    @pytest.mark.integration
    async def test_connection_survives_idle_after_audio(
        self, stt_short_keepalive, mia_audio_16khz, participant
    ):
        """WS must stay open across an idle window after real audio has been sent.

        Exercises the keep-alive path: once a real chunk sets the queue's
        sample_rate, ``get_samples`` raises ``QueueEmpty`` after 100 ms instead
        of letting ``wait_for`` time out. Without the fix the silence frame
        never fires and the server eventually closes the WS.
        """
        stt = stt_short_keepalive
        await stt.process_audio(mia_audio_16khz, participant=participant)
        await asyncio.sleep(stt.keepalive_interval_ms / 1000 * 3)
        await stt.process_audio(mia_audio_16khz, participant=participant)

        items = await stt.output.collect(timeout=15.0)
        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        assert len(finals) >= 2
