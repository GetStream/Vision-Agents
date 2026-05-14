import asyncio
import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins import inworld

load_dotenv()


class TestInworldSTT:
    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    async def stt(self):
        stt = inworld.STT()
        await stt.start()
        yield stt
        await stt.close()

    @pytest.mark.skipif(
        os.getenv("INWORLD_API_KEY") is None, reason="INWORLD_API_KEY not set"
    )
    @pytest.mark.integration
    async def test_transcribe_mia_audio(
        self, stt, mia_audio_48khz, silence_2s_48khz, participant
    ):
        for chunk in mia_audio_48khz.chunks(480):
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.001)

        for chunk in silence_2s_48khz.chunks(480):
            await stt.process_audio(chunk, participant=participant)
            await asyncio.sleep(0.001)

        items = await stt.output.collect(timeout=10.0)
        transcripts = [i for i in items if isinstance(i, Transcript)]
        finals = [t for t in transcripts if t.final]
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
        assert transcripts[0].participant == participant

        turn_starts = [i for i in items if isinstance(i, TurnStarted)]
        turn_ends = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turn_starts) >= 1
        assert len(turn_ends) >= 1
        assert turn_starts[0].participant == participant

    async def test_requires_api_key(self):
        env_backup = os.environ.pop("INWORLD_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="INWORLD_API_KEY"):
                inworld.STT(api_key=None)
        finally:
            if env_backup is not None:
                os.environ["INWORLD_API_KEY"] = env_backup

    async def test_handle_speech_started_emits_turn(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        stt._handle_message({"result": {"speechStarted": {"confidence": 0.95}}})

        turn_starts = [i for i in stt.output.peek() if isinstance(i, TurnStarted)]
        assert len(turn_starts) == 1
        assert turn_starts[0].participant == participant
        assert turn_starts[0].confidence == 0.95

    async def test_handle_speech_started_emits_every_time(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        stt._handle_message({"result": {"speechStarted": {"confidence": 0.9}}})
        stt._handle_message({"result": {"speechStarted": {"confidence": 0.8}}})

        turn_starts = [i for i in stt.output.peek() if isinstance(i, TurnStarted)]
        assert len(turn_starts) == 2

    async def test_handle_speech_stopped_emits_turn_with_silence(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant
        stt._speaking = True

        stt._handle_message({"result": {"speechStopped": {"silenceDurationMs": 750}}})

        turn_ends = [i for i in stt.output.peek() if isinstance(i, TurnEnded)]
        assert len(turn_ends) == 1
        assert turn_ends[0].trailing_silence_ms == 750.0

    async def test_handle_interim_transcript_is_replacement(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        stt._handle_message(
            {"result": {"transcription": {"transcript": "Hello", "isFinal": False}}}
        )

        transcripts = [i for i in stt.output.peek() if isinstance(i, Transcript)]
        assert len(transcripts) == 1
        assert transcripts[0].text == "Hello"
        assert transcripts[0].mode == "replacement"
        assert transcripts[0].participant == participant

    async def test_handle_final_transcript_ends_turn(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant
        stt._speaking = True

        stt._handle_message(
            {
                "result": {
                    "transcription": {"transcript": "Hello world", "isFinal": True}
                }
            }
        )

        items = stt.output.peek()
        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert len(transcripts) == 1
        assert transcripts[0].text == "Hello world"
        assert transcripts[0].mode == "final"

        turn_ends = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turn_ends) == 1

    async def test_speech_stopped_then_final_emits_single_turn_ended(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant
        stt._speaking = True

        stt._handle_message({"result": {"speechStopped": {"silenceDurationMs": 500}}})
        stt._handle_message(
            {"result": {"transcription": {"transcript": "Hi", "isFinal": True}}}
        )

        items = stt.output.peek()
        turn_ends = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turn_ends) == 1

    async def test_empty_interim_ignored(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        stt._handle_message(
            {"result": {"transcription": {"transcript": "", "isFinal": False}}}
        )

        assert stt.output.peek() == []

    async def test_empty_final_still_ends_turn(self):
        stt = inworld.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant
        stt._speaking = True

        stt._handle_message(
            {"result": {"transcription": {"transcript": "", "isFinal": True}}}
        )

        items = stt.output.peek()
        assert [i for i in items if isinstance(i, Transcript)] == []
        turn_ends = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turn_ends) == 1

    async def test_message_without_participant_emits_nothing(self):
        stt = inworld.STT(api_key="test-key")
        stt._current_participant = None

        stt._handle_message(
            {"result": {"transcription": {"transcript": "Hello", "isFinal": True}}}
        )

        assert stt.output.peek() == []

    async def test_clear_resets_turn_state(self):
        stt = inworld.STT(api_key="test-key")
        stt._speaking = True
        stt._audio_start_time = 123.0
        stt._audio_buffer.extend(b"\x00" * 100)

        await stt.clear()

        assert stt._speaking is False
        assert stt._audio_start_time is None
        assert len(stt._audio_buffer) == 0
