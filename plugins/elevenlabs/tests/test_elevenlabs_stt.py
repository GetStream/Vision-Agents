import asyncio

import pytest
from dotenv import load_dotenv

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins import elevenlabs

load_dotenv()


class TestElevenLabsSTTCallbacks:
    """Unit coverage for translating provider callbacks into turn events."""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    def stt(self, participant: Participant) -> elevenlabs.STT:
        stt = elevenlabs.STT(api_key="test-key")
        stt._current_participant = participant
        return stt

    def test_duplicate_partials_emit_one_turn_started(
        self, stt: elevenlabs.STT
    ) -> None:
        stt._on_partial_transcript({"text": "Hello"})
        stt._on_partial_transcript({"text": "Hello world"})

        turn_events = [
            item for item in stt.output.peek() if isinstance(item, TurnStarted)
        ]
        assert len(turn_events) == 1

    def test_multiple_committed_utterances_emit_balanced_ordered_turns(
        self, stt: elevenlabs.STT
    ) -> None:
        stt._on_partial_transcript({"text": "First"})
        stt._on_committed_transcript({"text": "First utterance"})
        stt._on_partial_transcript({"text": "Second"})
        stt._on_committed_transcript({"text": "Second utterance"})

        turn_events = [
            item
            for item in stt.output.peek()
            if isinstance(item, (TurnStarted, TurnEnded))
        ]
        assert len(turn_events) == 4
        assert all(
            isinstance(turn_events[index], TurnStarted)
            and isinstance(turn_events[index + 1], TurnEnded)
            for index in range(0, len(turn_events), 2)
        )

    def test_empty_commit_ends_active_turn(self, stt: elevenlabs.STT) -> None:
        stt._on_partial_transcript({"text": "Speech"})
        stt._on_committed_transcript({"text": ""})

        turn_events = [
            item
            for item in stt.output.peek()
            if isinstance(item, (TurnStarted, TurnEnded))
        ]
        assert len(turn_events) == 2
        assert isinstance(turn_events[0], TurnStarted)
        assert isinstance(turn_events[1], TurnEnded)

    def test_keepalive_empty_commit_does_not_emit_turn_events(
        self, stt: elevenlabs.STT
    ) -> None:
        stt._on_committed_transcript({"text": ""})
        stt._on_committed_transcript({"text": "   "})

        assert not any(
            isinstance(item, (TurnStarted, TurnEnded)) for item in stt.output.peek()
        )


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
        """Every provider-detected utterance has ordered start and end events."""
        await stt.process_audio(mia_audio_16khz, participant=participant)

        items = await stt.output.collect(timeout=10.0)
        turn_events = [
            item for item in items if isinstance(item, (TurnStarted, TurnEnded))
        ]
        turn_started = [item for item in turn_events if isinstance(item, TurnStarted)]
        turn_ended = [item for item in turn_events if isinstance(item, TurnEnded)]

        assert turn_started
        assert len(turn_started) == len(turn_ended)
        assert all(
            isinstance(turn_events[index], TurnStarted)
            and isinstance(turn_events[index + 1], TurnEnded)
            for index in range(0, len(turn_events), 2)
        )
        assert all(event.participant == participant for event in turn_events)

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
