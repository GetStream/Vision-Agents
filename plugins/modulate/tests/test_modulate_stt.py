import asyncio

import pytest
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded
from vision_agents.plugins import modulate


class TestModulateSTT:
    """Tests for Modulate AI STT."""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self, mia_audio_48khz, silence_2s_48khz, participant
    ):
        stt = modulate.STT()
        await stt.start()
        try:
            for chunk in mia_audio_48khz.chunks(480):
                await stt.process_audio(chunk, participant=participant)
                await asyncio.sleep(0.001)

            for chunk in silence_2s_48khz.chunks(480):
                await stt.process_audio(chunk, participant=participant)
                await asyncio.sleep(0.001)

            items = await stt.output.collect(timeout=10.0)
        finally:
            await stt.close()

        transcripts = [i for i in items if isinstance(i, Transcript)]
        finals = [t for t in transcripts if t.final]
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
        assert transcripts[0].participant == participant

    async def test_utterance_emits_final_transcript_and_turn_ended(self, participant):
        # Drives _handle_utterance directly rather than through process_audio: the
        # public path requires a live WebSocket, and the project rule is to never
        # mock. This asserts on the output stream, but does set _current_participant.
        stt = modulate.STT(api_key="test-key")
        stt._current_participant = participant

        stt._handle_utterance(
            {
                "utterance_uuid": "abc-123",
                "text": "Hello world",
                "duration_ms": 2000,
                "speaker": 1,
                "language": "en",
            }
        )

        items = list(stt.output.peek())
        transcripts = [i for i in items if isinstance(i, Transcript)]
        turn_ends = [i for i in items if isinstance(i, TurnEnded)]

        assert len(transcripts) == 1
        assert transcripts[0].text == "Hello world"
        assert transcripts[0].final
        assert transcripts[0].response.language == "en"
        assert transcripts[0].participant.user_id == "speaker_1"
        assert len(turn_ends) == 1

    async def test_extra_kwargs_forwarded_to_ws_url(self):
        stt = modulate.STT(api_key="test-key", custom_flag=True, region="us-east")
        url = stt._build_ws_url()
        assert "custom_flag=true" in url
        assert "region=us-east" in url
