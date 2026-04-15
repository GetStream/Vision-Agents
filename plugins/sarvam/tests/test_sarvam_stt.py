"""Tests for the Sarvam STT plugin."""

import json
import os

import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.plugins.sarvam import STT

from conftest import STTSession

load_dotenv()


class TestSarvamSTT:
    """Unit tests for Sarvam STT configuration."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            STT()

    async def test_default_configuration(self):
        stt = STT(api_key="sk_test")
        assert stt.model == "saaras:v3"
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

    async def test_build_ws_url_includes_query_params(self):
        stt = STT(
            api_key="sk_test",
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
        assert "sample_rate=16000" in url

    async def test_build_ws_url_without_language(self):
        stt = STT(api_key="sk_test")
        url = stt._build_ws_url()
        assert "language-code" not in url

    async def test_build_ws_url_translate_endpoint_for_saaras_v25(self):
        stt = STT(api_key="sk_test", model="saaras:v2.5")
        url = stt._build_ws_url()
        assert url.startswith("wss://api.sarvam.ai/speech-to-text-translate/ws?")

    async def test_build_ws_url_regular_endpoint_for_saarika(self):
        stt = STT(api_key="sk_test", model="saarika:v2.5")
        url = stt._build_ws_url()
        assert url.startswith("wss://api.sarvam.ai/speech-to-text/ws?")

    async def test_mode_only_included_for_supporting_models(self):
        stt = STT(api_key="sk_test", model="saarika:v2.5", mode="transcribe")
        url = stt._build_ws_url()
        assert "mode=" not in url

        stt_v3 = STT(api_key="sk_test", model="saaras:v3", mode="translate")
        url_v3 = stt_v3._build_ws_url()
        assert "mode=translate" in url_v3

    async def test_process_audio_uses_pcm_s16le_codec(self):
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
        assert message["audio"]["encoding"] == "audio/wav"
        assert message["audio"]["sample_rate"] == 16000


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
        session = STTSession(stt)

        await stt.process_audio(
            mia_audio_48khz, participant=Participant({}, user_id="hi", id="hi")
        )
        await stt.process_audio(
            silence_2s_48khz, participant=Participant({}, user_id="hi", id="hi")
        )

        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert full_transcript is not None
        normalized = full_transcript.lower().strip()
        assert normalized
        assert any(keyword in normalized for keyword in ("forgotten", "treasure"))
