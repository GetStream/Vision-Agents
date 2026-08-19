"""Tests for the Sarvam STT plugin."""

import asyncio
import json
import os

import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins.sarvam import STT

load_dotenv()


class TestSarvamSTT:
    """Unit tests for Sarvam STT configuration."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            STT()

    async def test_default_configuration(self):
        stt = STT(api_key="sk_test")
        assert stt.model == "saaras:v3-realtime"
        assert stt.sample_rate == 16000
        assert stt.vad_signals is True
        assert stt.turn_detection is True
        assert stt.provider_name == "sarvam"

    async def test_invalid_model_rejected(self):
        with pytest.raises(ValueError, match="Unsupported Sarvam STT model"):
            STT(api_key="sk_test", model="not-a-model")

    async def test_invalid_sample_rate_rejected(self):
        with pytest.raises(ValueError, match="sample_rate"):
            STT(api_key="sk_test", sample_rate=44100)

    async def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            STT(api_key="sk_test", mode="not-a-mode")

    async def test_realtime_ws_url_includes_query_params(self):
        stt = STT(
            api_key="sk_test",
            language="hi-IN",
            mode="translate",
            prompt="medical terms",
        )
        url = stt._build_ws_url()
        assert url.startswith("wss://api.sarvam.ai/speech-to-text-realtime/ws?")
        assert "model=saaras%3Av3-realtime" in url
        assert "language_code=hi-IN" in url
        assert "mode=translate" in url
        assert "encoding=linear16" in url
        assert "stream_type=fast" in url
        assert "endpointing=vad" in url
        assert "prompt=medical+terms" in url
        assert "sample_rate=16000" in url
        assert "language-code" not in url
        assert "vad_signals" not in url
        assert "high_vad_sensitivity" not in url

    async def test_realtime_ws_url_defaults_language_to_auto(self):
        stt = STT(api_key="sk_test")
        url = stt._build_ws_url()
        assert "language_code=auto" in url

    async def test_legacy_models_rejected(self):
        for model in ("saarika:v2.5", "saaras:v2", "saaras:v2.5"):
            with pytest.raises(ValueError, match="Unsupported Sarvam STT model"):
                STT(api_key="sk_test", model=model)

    async def test_legacy_v3_ws_url(self):
        stt = STT(
            api_key="sk_test",
            model="saaras:v3",
            language="hi-IN",
            mode="translate",
            high_vad_sensitivity=True,
        )
        url = stt._build_ws_url()
        assert url.startswith("wss://api.sarvam.ai/speech-to-text/ws?")
        assert "model=saaras%3Av3" in url
        assert "language-code=hi-IN" in url
        assert "mode=translate" in url
        assert "vad_signals=true" in url
        assert "high_vad_sensitivity=true" in url
        assert "language_code" not in url

    async def test_process_audio_sends_realtime_audio_input(self):
        class FakeWebSocket:
            def __init__(self) -> None:
                self.closed = False
                self.sent_messages: list[str] = []

            async def send_str(self, message: str) -> None:
                self.sent_messages.append(message)

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

        message = json.loads(ws.sent_messages[0])
        assert message["event"] == "audio_input"
        assert isinstance(message["audio"], str)

    async def test_process_audio_legacy_uses_pcm_s16le_codec(self):
        class FakeWebSocket:
            def __init__(self) -> None:
                self.closed = False
                self.sent_messages: list[str] = []

            async def send_str(self, message: str) -> None:
                self.sent_messages.append(message)

        stt = STT(api_key="sk_test", model="saaras:v3")
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

        message = json.loads(ws.sent_messages[0])
        assert message["audio"]["encoding"] == "audio/wav"
        assert message["audio"]["sample_rate"] == 16000

    async def test_realtime_events_emit_turns_and_transcripts(self):
        stt = STT(api_key="sk_test")
        participant = Participant({}, user_id="user-1", id="user-1")
        stt._current_participant = participant

        stt._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
        stt._handle_message(
            {
                "event": "transcript.partial",
                "utterance_idx": 0,
                "text": "hello",
                "language": "en-IN",
            }
        )
        stt._handle_message(
            {
                "event": "transcript.final",
                "utterance_idx": 0,
                "text": "hello there",
                "language": "en-IN",
            }
        )
        stt._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

        items = await stt.output.collect(timeout=0)
        assert isinstance(items[0], TurnStarted)
        assert isinstance(items[1], Transcript) and items[1].mode == "replacement"
        assert items[1].text == "hello"
        assert isinstance(items[2], Transcript) and items[2].final
        assert items[2].text == "hello there"
        assert isinstance(items[3], TurnEnded)


@pytest.mark.skipif(not os.getenv("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set")
@pytest.mark.integration
class TestSarvamSTTIntegration:
    """Integration tests against the real Sarvam streaming STT."""

    @pytest.fixture
    async def stt(self):
        s = STT(language="en-IN")
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
