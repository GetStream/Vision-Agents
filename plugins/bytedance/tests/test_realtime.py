import asyncio

import pytest
from dotenv import load_dotenv

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.realtime import (
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeUserTranscript,
)
from vision_agents.plugins import bytedance
from vision_agents.plugins.bytedance import _ast

load_dotenv()


def _response_bytes(event: int, *, text: str = "", data: bytes = b"") -> bytes:
    out = _ast._int_field(2, event)
    if data:
        out += _ast._bytes_field(3, data)
    if text:
        out += _ast._string_field(4, text)
    return out


class TestBytedanceRealtime:
    @pytest.fixture
    def rt(self) -> bytedance.Realtime:
        return bytedance.Realtime(
            source_language="zh", target_language="en", api_key="test-key"
        )

    def test_requires_zh_or_en(self):
        with pytest.raises(ValueError):
            bytedance.Realtime(
                source_language="ja", target_language="ko", api_key="test-key"
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            bytedance.Realtime(
                source_language="zh",
                target_language="en",
                mode="chat",
                api_key="test-key",
            )

    def test_start_session_request_has_language_pair(self, rt):
        request = _ast.TranslateRequest(
            event=100,
            session_id=rt.session_id,
            request=_ast.ReqParams(
                mode="s2s", source_language="zh", target_language="en"
            ),
        )
        # The plugin's own builder must produce a decodable request with the pair.
        encoded = rt._start_session_request()
        assert request.request.encode() in encoded

    def test_source_subtitle_emits_user_transcript(self, rt):
        rt._current_participant = Participant({}, user_id="u", id="u")

        rt._handle_response(_response_bytes(652, text="你好"))

        transcripts = [
            i for i in rt.output.peek() if isinstance(i, RealtimeUserTranscript)
        ]
        assert [t.text for t in transcripts] == ["你好"]
        assert transcripts[0].mode == "final"

    def test_translation_subtitle_emits_agent_transcript(self, rt):
        rt._handle_response(_response_bytes(655, text="hello"))

        transcripts = [
            i for i in rt.output.peek() if isinstance(i, RealtimeAgentTranscript)
        ]
        assert [t.text for t in transcripts] == ["hello"]
        assert transcripts[0].mode == "final"

    def test_tts_response_emits_audio_output(self, rt):
        pcm_bytes = b"\x00\x01" * 240
        rt._handle_response(_response_bytes(352, data=pcm_bytes))

        audio = [i for i in rt.output.peek() if isinstance(i, RealtimeAudioOutput)]
        assert len(audio) == 1
        assert audio[0].data.sample_rate == 24000

    def test_session_started_event_sets_gate(self, rt):
        assert not rt._session_started
        rt._handle_response(_response_bytes(150))
        assert rt._session_started

    @pytest.mark.integration
    async def test_interpretation_flow(self, mia_audio_16khz, silence_1s_16khz):
        rt = bytedance.Realtime(source_language="en", target_language="zh")
        participant = Participant({}, user_id="u", id="u")
        try:
            await rt.connect()
            await asyncio.sleep(2.0)
            await rt.process_audio(silence_1s_16khz, participant)
            await rt.process_audio(mia_audio_16khz, participant)
            await asyncio.sleep(12.0)
            items = rt.output.peek()
        finally:
            await rt.close()

        assert any(isinstance(i, RealtimeAudioOutput) for i in items)
