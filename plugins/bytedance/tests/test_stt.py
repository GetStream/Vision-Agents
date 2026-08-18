import gzip
import json

import pytest
from dotenv import load_dotenv

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded
from vision_agents.plugins import bytedance
from vision_agents.plugins.bytedance import _v3

load_dotenv()


def _server_result_frame(result: dict) -> _v3.Message:
    body = gzip.compress(json.dumps({"result": result}).encode())
    header = bytes(
        [
            (0b0001 << 4) | 0b0001,
            (int(_v3.MsgType.FULL_SERVER_RESPONSE) << 4) | _v3.Flags.POS_SEQ,
            (int(_v3.Serialization.JSON) << 4) | int(_v3.Compression.GZIP),
            0,
        ]
    )
    frame = (
        header
        + (1).to_bytes(4, "big", signed=True)
        + len(body).to_bytes(4, "big")
        + body
    )
    return _v3.parse_response(frame)


class TestBytedanceSTT:
    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    def stt(self) -> bytedance.STT:
        return bytedance.STT(api_key="test-key")

    def test_partial_result_emits_replacement(self, stt, participant):
        stt._current_participant = participant

        stt._handle_message(_server_result_frame({"text": "hello wor"}))

        items = stt.output.peek()
        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert len(transcripts) == 1
        assert transcripts[0].mode == "replacement"
        assert transcripts[0].text == "hello wor"

    def test_definite_utterance_emits_final_and_turn_ended(self, stt, participant):
        stt._current_participant = participant

        stt._handle_message(
            _server_result_frame(
                {
                    "text": "hello world",
                    "utterances": [{"text": "hello world", "definite": True}],
                }
            )
        )

        items = stt.output.peek()
        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        assert [t.text for t in finals] == ["hello world"]
        assert any(isinstance(i, TurnEnded) for i in items)

    async def test_error_frame_does_not_emit_transcript(self, stt, participant):
        stt._current_participant = participant
        error_message = _v3.Message(type=_v3.MsgType.ERROR, code=45000000, payload={})

        stt._handle_message(error_message)

        assert stt.output.peek() == []

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BYTEDANCE_API_KEY", raising=False)
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("BYTEDANCE_APP_KEY", raising=False)
        monkeypatch.delenv("BYTEDANCE_ACCESS_KEY", raising=False)
        with pytest.raises(ValueError):
            bytedance.STT()

    @pytest.mark.integration
    async def test_transcribe_mia_audio(self, mia_audio_16khz, participant):
        stt = bytedance.STT()
        try:
            await stt.start()
            await stt.process_audio(mia_audio_16khz, participant=participant)
            items = await stt.output.collect(timeout=15.0)
        finally:
            await stt.close()

        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        full = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full.lower()
