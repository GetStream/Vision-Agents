import os

import pytest
from dotenv import load_dotenv

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins import cartesia

load_dotenv()


class TestCartesiaSTT:
    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    def test_defaults_to_latest_model_and_turn_detection(self):
        stt = cartesia.STT(api_key="fake")

        assert stt.model == "ink-2"
        assert stt.turn_detection is True
        assert stt.eager_turn_detection is True
        assert "model=ink-2" in stt._build_websocket_url()
        assert "cartesia_version=2026-03-01" in stt._build_websocket_url()

    async def test_handle_turn_events(self, participant):
        stt = cartesia.STT(api_key="fake")
        stt._current_participant = participant

        stt._handle_message({"type": "turn.start", "confidence": 0.8})
        stt._handle_message(
            {
                "type": "turn.update",
                "transcript": "hello",
                "words": [{"word": "hello", "confidence": 0.9}],
                "duration": 0.5,
            }
        )
        stt._handle_message(
            {
                "type": "turn.eager_end",
                "transcript": "hello world",
                "confidence": 0.7,
            }
        )
        stt._handle_message({"type": "turn.resume", "confidence": 0.6})
        stt._handle_message(
            {
                "type": "turn.end",
                "transcript": "hello world again",
                "confidence": 0.95,
                "duration_ms": 1200,
                "trailing_silence_ms": 250,
            }
        )

        items = await stt.output.collect(timeout=0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert [t.text for t in transcripts] == [
            "hello",
            "hello world",
            "hello world again",
        ]
        assert [t.final for t in transcripts] == [False, False, True]
        assert transcripts[0].confidence == 0.9
        assert transcripts[-1].model_name == "ink-2"

        turns_started = [i for i in items if isinstance(i, TurnStarted)]
        turns_ended = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turns_started) == 2
        assert turns_started[0].confidence == 0.8
        assert turns_started[1].confidence == 0.6
        assert [t.eager for t in turns_ended] == [True, False]
        assert turns_ended[-1].duration_ms == 1200
        assert turns_ended[-1].trailing_silence_ms == 250

    @pytest.mark.skipif(
        os.getenv("CARTESIA_API_KEY") is None, reason="CARTESIA_API_KEY not set"
    )
    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self, mia_audio_48khz, silence_2s_48khz, participant
    ):
        stt = cartesia.STT()
        await stt.start()
        try:
            await stt.process_audio(mia_audio_48khz, participant=participant)
            await stt.process_audio(silence_2s_48khz, participant=participant)

            items = await stt.output.collect(timeout=10.0)
        finally:
            await stt.close()

        transcripts = [i for i in items if isinstance(i, Transcript)]
        finals = [t for t in transcripts if t.final]
        assert finals, "No final Transcript emitted by Cartesia STT"
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
        assert any(isinstance(i, TurnEnded) for i in items)
