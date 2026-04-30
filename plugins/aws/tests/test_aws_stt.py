"""Integration tests for AWS Transcribe STT."""

import pytest
from dotenv import load_dotenv
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
from vision_agents.plugins import aws

from conftest import STTSession

load_dotenv()


@pytest.mark.integration
class TestTranscribeSTT:
    """Integration tests against real AWS Transcribe streaming."""

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
            word in full_transcript
            for word in ["village", "quiet", "mia", "treasures"]
        ), f"Transcript did not match expected content: {full_transcript!r}"

    async def test_partial_transcripts_emitted(
        self, stt, mia_audio_16khz_chunked, participant
    ):
        session = STTSession(stt)

        for chunk in mia_audio_16khz_chunked:
            await stt.process_audio(chunk, participant=participant)

        await session.wait_for_result(timeout=30.0)
        assert session.partial_transcripts, "No partial transcripts received"

    async def test_turn_events_emitted(
        self, stt, mia_audio_16khz_chunked, participant
    ):
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
