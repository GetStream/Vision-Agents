"""Tests for the Smallest AI STT plugin."""

import asyncio
import os

import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.stt.events import STTErrorEvent
from vision_agents.plugins.smallest import STT

load_dotenv()


class TestSmallestSTT:
    """Unit tests for Smallest AI STT configuration."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SMALLEST_API_KEY"):
            STT()

    async def test_default_configuration(self):
        stt = STT(api_key="sk_test")
        assert stt.model == "pulse"
        assert stt.language == "en"
        assert stt.sample_rate == 16000
        assert stt.word_timestamps is False
        assert stt.provider_name == "smallest"

    async def test_ws_url_includes_query_params(self):
        stt = STT(api_key="sk_test", language="hi", sample_rate=8000)
        url = stt._build_ws_url()
        assert url.startswith("wss://api.smallest.ai/waves/v1/stt/live?")
        assert "model=pulse" in url
        assert "language=hi" in url
        assert "encoding=linear16" in url
        assert "sample_rate=8000" in url
        assert "word_timestamps=false" in url

    async def test_process_audio_sends_binary_frame(self):
        class FakeWebSocket:
            def __init__(self) -> None:
                self.closed = False
                self.sent_bytes: list[bytes] = []

            async def send_bytes(self, data: bytes) -> None:
                self.sent_bytes.append(data)

        stt = STT(api_key="sk_test")
        ws = FakeWebSocket()
        stt._ws = ws
        stt._connection_ready.set()

        pcm_data = PcmData.from_bytes(
            b"\x01\x00" * 160,
            sample_rate=16000,
            channels=1,
        )
        participant = Participant({}, user_id="user-1", id="user-1")

        await stt.process_audio(pcm_data, participant=participant)

        assert len(ws.sent_bytes) == 1
        assert stt._current_participant == participant

    async def test_handle_message_emits_transcripts(self):
        stt = STT(api_key="sk_test")
        participant = Participant({}, user_id="user-1", id="user-1")
        stt._current_participant = participant

        stt._handle_message(
            {"transcript": "hello", "is_final": False, "language": "en"}
        )
        stt._handle_message(
            {"transcript": "hello there", "is_final": True, "language": "en"}
        )

        items = await stt.output.collect(timeout=0)
        assert isinstance(items[0], Transcript) and items[0].mode == "replacement"
        assert items[0].text == "hello"
        assert isinstance(items[1], Transcript) and items[1].final
        assert items[1].text == "hello there"

    async def test_handle_message_ignores_is_last(self):
        stt = STT(api_key="sk_test")
        participant = Participant({}, user_id="user-1", id="user-1")
        stt._current_participant = participant

        stt._handle_message({"transcript": "", "is_final": True, "is_last": True})

        items = await stt.output.collect(timeout=0)
        assert items == []

    async def test_handle_message_emits_error(self):
        stt = STT(api_key="sk_test")

        errors = []

        @stt.events.subscribe
        async def _on_error(event: STTErrorEvent):
            errors.append(event)

        stt._handle_message({"status": "error", "message": "boom"})
        await stt.events.wait()

        assert len(errors) == 1
        assert "boom" in str(errors[0].error)


@pytest.mark.skipif(
    not os.getenv("SMALLEST_API_KEY"), reason="SMALLEST_API_KEY not set"
)
@pytest.mark.integration
class TestSmallestSTTIntegration:
    """Integration tests against the real Smallest AI streaming STT."""

    @pytest.fixture
    async def stt(self):
        s = STT(language="en")
        try:
            await s.start()
            yield s
        finally:
            await s.close()

    async def test_transcribe_mia_audio_48khz(
        self, stt, mia_audio_48khz, silence_2s_48khz
    ):
        participant = Participant({}, user_id="id", id="id")
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
