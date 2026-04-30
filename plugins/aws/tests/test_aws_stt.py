import asyncio

import pytest
from aws_sdk_transcribe_streaming.models import (
    Alternative,
    Result,
    Transcript,
    TranscriptEvent,
)
from dotenv import load_dotenv
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
from vision_agents.plugins import aws

from conftest import STTSession

load_dotenv()


class TestTranscribeSTT:
    @pytest.fixture
    def transcript_event_factory(self):
        def factory(
            text: str, *, is_partial: bool, start_time: float
        ) -> TranscriptEvent:
            return TranscriptEvent(
                transcript=Transcript(
                    results=[
                        Result(
                            result_id="r",
                            start_time=start_time,
                            end_time=start_time + 1.0,
                            is_partial=is_partial,
                            alternatives=[Alternative(transcript=text, items=[])],
                        )
                    ]
                )
            )

        return factory

    async def test_partial_result_emits_partial_transcript_and_turn_started(
        self, participant, transcript_event_factory
    ):
        stt = aws.STT(language_code="en-US")
        stt._current_participant = participant
        session = STTSession(stt)
        turn_started: list[TurnStartedEvent] = []

        @stt.events.subscribe
        async def on_turn_started(event: TurnStartedEvent):
            turn_started.append(event)

        stt._handle_transcript_event(
            transcript_event_factory("hello", is_partial=True, start_time=0.0)
        )
        await asyncio.sleep(0.05)

        assert [e.text for e in session.partial_transcripts] == ["hello"]
        assert not session.transcripts
        assert len(turn_started) == 1
        assert turn_started[0].participant == participant

    async def test_final_result_emits_transcript_and_turn_ended(
        self, participant, transcript_event_factory
    ):
        stt = aws.STT(language_code="en-US")
        stt._current_participant = participant
        session = STTSession(stt)
        turn_ended: list[TurnEndedEvent] = []

        @stt.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            turn_ended.append(event)

        stt._handle_transcript_event(
            transcript_event_factory("hello world", is_partial=False, start_time=0.0)
        )
        await asyncio.sleep(0.05)

        assert [e.text for e in session.transcripts] == ["hello world"]
        assert len(turn_ended) == 1
        assert turn_ended[0].participant == participant

    async def test_clear_drops_results_before_watermark(
        self, participant, transcript_event_factory
    ):
        stt = aws.STT(language_code="en-US")
        stt._audio_sent_seconds = 5.0
        await stt.clear()

        stt._current_participant = participant
        session = STTSession(stt)

        stt._handle_transcript_event(
            transcript_event_factory("stale", is_partial=False, start_time=2.0)
        )
        stt._handle_transcript_event(
            transcript_event_factory("fresh", is_partial=False, start_time=6.0)
        )
        await asyncio.sleep(0.05)

        assert [e.text for e in session.transcripts] == ["fresh"]


@pytest.mark.integration
class TestTranscribeSTTIntegration:
    @pytest.fixture
    async def stt(self):
        stt = aws.STT(language_code="en-US")
        try:
            await stt.start()
            yield stt
        finally:
            await stt.close()

    async def test_transcribe_mia_audio_16khz(
        self, stt, mia_audio_16khz_chunked, participant
    ):
        session = STTSession(stt)

        for chunk in mia_audio_16khz_chunked:
            await stt.process_audio(chunk, participant=participant)

        await session.wait_for_result(timeout=30.0)
        assert not session.errors, f"Errors occurred: {session.errors}"

        full_transcript = session.get_full_transcript().lower()
        assert any(
            word in full_transcript for word in ["village", "quiet", "mia", "treasures"]
        ), f"Transcript did not match expected content: {full_transcript!r}"

    async def test_partial_transcripts_emitted(
        self, stt, mia_audio_16khz_chunked, participant
    ):
        session = STTSession(stt)

        for chunk in mia_audio_16khz_chunked:
            await stt.process_audio(chunk, participant=participant)

        await session.wait_for_result(timeout=30.0)
        assert session.partial_transcripts, "No partial transcripts received"

    async def test_turn_events_emitted(self, stt, mia_audio_16khz_chunked, participant):
        session = STTSession(stt)
        turn_started: list[TurnStartedEvent] = []
        turn_ended: list[TurnEndedEvent] = []

        @stt.events.subscribe
        async def on_turn_started(event: TurnStartedEvent):
            turn_started.append(event)

        @stt.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            turn_ended.append(event)

        for chunk in mia_audio_16khz_chunked:
            await stt.process_audio(chunk, participant=participant)

        await session.wait_for_result(timeout=30.0)

        assert turn_started, "No TurnStartedEvent received"
        assert turn_ended, "No TurnEndedEvent received"
        assert turn_started[0].participant == participant
        assert turn_ended[0].participant == participant
