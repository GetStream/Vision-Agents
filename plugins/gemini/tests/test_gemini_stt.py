import asyncio
import os

import pytest
import vision_agents.plugins.gemini.stt as gemini_stt_module
from google.genai.types import (
    LiveServerContent,
    LiveServerMessage,
    Transcription,
    WordInfo,
)
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins import gemini
from vision_agents.plugins.gemini.stt import DEFAULT_MODEL


class TestGeminiSTT:
    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    def stt(self) -> gemini.STT:
        return gemini.STT(api_key="fake")

    def test_configuration(self):
        stt = gemini.STT(
            api_key="fake",
            language_codes=["en-US"],
            custom_vocabulary=["Vision Agents"],
        )

        assert stt.model == DEFAULT_MODEL
        assert stt.turn_detection
        assert stt._config["input_audio_transcription"] == {
            "language_codes": ["en-US"],
            "custom_vocabulary": ["Vision Agents"],
        }

    def test_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
            gemini.STT()

    async def test_interim_and_final_transcripts(self, stt, participant):
        stt._current_participant = participant
        stt._audio_duration_ms = 1200

        stt._handle_message(
            LiveServerMessage(
                server_content=LiveServerContent(
                    interim_input_transcription=Transcription(
                        text="hello",
                        language_code="en-US",
                    )
                )
            )
        )
        stt._handle_message(
            LiveServerMessage(
                server_content=LiveServerContent(
                    input_transcription=Transcription(
                        text="hello world",
                        finished=True,
                        language_code="en-US",
                        speaker_label="spk_1",
                        words=[
                            WordInfo(
                                word="hello",
                                start_offset="0s",
                                end_offset="0.5s",
                            )
                        ],
                    )
                )
            )
        )

        items = await stt.output.collect(timeout=0)
        transcripts = [item for item in items if isinstance(item, Transcript)]

        assert [item.mode for item in transcripts] == ["replacement", "final"]
        assert [item.text for item in transcripts] == ["hello", "hello world"]
        assert transcripts[-1].language == "en-US"
        assert transcripts[-1].audio_duration_ms == 1200
        assert transcripts[-1].response.other == {
            "speaker_label": "spk_1",
            "words": [
                {
                    "word": "hello",
                    "start_offset": "0s",
                    "end_offset": "0.5s",
                }
            ],
        }
        assert len([item for item in items if isinstance(item, TurnStarted)]) == 1
        assert len([item for item in items if isinstance(item, TurnEnded)]) == 1

    async def test_delta_transcripts_finish_on_turn_complete(self, stt, participant):
        stt._current_participant = participant

        stt._handle_message(
            LiveServerMessage(
                server_content=LiveServerContent(
                    input_transcription=Transcription(text="forgotten ")
                )
            )
        )
        stt._handle_message(
            LiveServerMessage(
                server_content=LiveServerContent(
                    input_transcription=Transcription(text="treasures"),
                    turn_complete=True,
                )
            )
        )

        items = await stt.output.collect(timeout=0)
        transcripts = [item for item in items if isinstance(item, Transcript)]

        assert [item.mode for item in transcripts] == ["delta", "delta", "final"]
        assert transcripts[-1].text == "forgotten treasures"

    async def test_interim_transcript_finishes_after_updates_stop(
        self,
        stt,
        participant,
        monkeypatch,
    ):
        monkeypatch.setattr(
            gemini_stt_module,
            "FINAL_TRANSCRIPT_DELAY_SECONDS",
            0.01,
        )
        stt._current_participant = participant
        stt._handle_message(
            LiveServerMessage(
                server_content=LiveServerContent(
                    interim_input_transcription=Transcription(text="complete thought")
                )
            )
        )

        await asyncio.sleep(0.02)

        items = await stt.output.collect(timeout=0)
        transcripts = [item for item in items if isinstance(item, Transcript)]
        assert [item.mode for item in transcripts] == ["replacement", "final"]
        assert transcripts[-1].text == "complete thought"
        assert any(isinstance(item, TurnEnded) for item in items)

    async def test_clear_and_close_reset_state(self, stt, participant):
        stt._current_participant = participant
        stt._handle_message(
            LiveServerMessage(
                server_content=LiveServerContent(
                    interim_input_transcription=Transcription(text="pending")
                )
            )
        )

        await stt.clear()

        assert await stt.output.collect(timeout=0) == []
        assert stt._current_participant is None
        assert not stt._turn_in_progress

        await stt.close()

        assert stt.closed
        assert stt.output.closed()

    @pytest.mark.integration
    async def test_transcribe_chunked_audio(
        self,
        mia_audio_16khz_chunked,
        silence_1s_16khz,
        participant,
    ):
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            pytest.fail(
                "Gemini integration tests require GOOGLE_API_KEY or GEMINI_API_KEY",
                pytrace=False,
            )

        stt = gemini.STT()
        await stt.start()
        try:
            for chunk in mia_audio_16khz_chunked:
                await stt.process_audio(chunk, participant)
                await asyncio.sleep(chunk.duration)
            for chunk in silence_1s_16khz.chunks(320):
                await stt.process_audio(chunk, participant)
                await asyncio.sleep(chunk.duration)

            items = await stt.output.collect(timeout=15.0)
        finally:
            await stt.close()

        transcripts = [
            item for item in items if isinstance(item, Transcript) and item.final
        ]
        assert transcripts, items[-5:]
        assert (
            "forgotten treasures" in " ".join(item.text for item in transcripts).lower()
        )
