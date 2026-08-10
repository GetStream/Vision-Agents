"""Tests for the Telnyx STT plugin."""

import asyncio
import os

import aiohttp
import numpy as np
import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.plugins.telnyx import STT

load_dotenv()


class TestTelnyxSTT:
    """Unit tests for Telnyx STT configuration and message handling."""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    def stt_with_failing_socket(self) -> STT:
        """An started STT whose socket rejects every send."""
        instance = STT(api_key="KEY_test")

        class FailingWS:
            closed = False

            async def send_bytes(self, data: bytes) -> None:
                raise aiohttp.ClientError("connection reset")

        instance.started = True
        instance._ws = FailingWS()
        instance._connection_ready.set()
        return instance

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TELNYX_API_KEY"):
            STT()

    async def test_default_configuration(self):
        stt = STT(api_key="KEY_test")
        assert stt.transcription_engine == "Telnyx"
        assert stt.language == "en"
        assert stt.sample_rate == 16000
        assert stt.interim_results is False
        assert stt.provider_name == "telnyx"
        assert stt.turn_detection is False

    async def test_unknown_engine_is_left_to_the_server(self):
        """The engine catalogue is served by Telnyx, so it is not pinned here."""
        stt = STT(api_key="KEY_test", transcription_engine="SomeNewEngine")
        assert "transcription_engine=SomeNewEngine" in stt._build_ws_url()

    async def test_url_carries_stream_parameters(self):
        stt = STT(api_key="KEY_test", language="es", sample_rate=8000)
        url = stt._build_ws_url()
        assert url.startswith("wss://api.telnyx.com/v2/speech-to-text/transcription?")
        assert "input_format=linear16" in url
        assert "sample_rate=8000" in url
        assert "language=es" in url
        assert "transcription_engine=Telnyx" in url

    async def test_url_uses_interim_results_not_partial_results(self):
        on = STT(api_key="KEY_test", interim_results=True)._build_ws_url()
        off = STT(api_key="KEY_test")._build_ws_url()
        assert "interim_results=true" in on
        assert "interim_results=false" in off
        assert "partial_results" not in on

    async def test_url_omits_model_when_unset(self):
        assert "model=" not in STT(api_key="KEY_test")._build_ws_url()
        assert (
            "model=whisper" in STT(api_key="KEY_test", model="whisper")._build_ws_url()
        )

    async def test_final_transcript_emitted(self, participant):
        stt = STT(api_key="KEY_test")
        stt._current_participant = participant

        stt._handle_message(
            {"transcript": "hello world", "confidence": 0.9, "is_final": True}
        )
        items = await stt.output.collect(timeout=0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert [t.text for t in transcripts] == ["hello world"]
        assert transcripts[0].final
        assert transcripts[0].confidence == 0.9
        assert transcripts[0].participant == participant

    async def test_interim_transcript_is_not_final(self, participant):
        stt = STT(api_key="KEY_test")
        stt._current_participant = participant

        stt._handle_message(
            {"transcript": "hello", "confidence": None, "is_final": False}
        )
        items = await stt.output.collect(timeout=0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert [t.text for t in transcripts] == ["hello"]
        assert not transcripts[0].final
        assert transcripts[0].confidence is None

    async def test_empty_transcript_emits_nothing(self, participant):
        stt = STT(api_key="KEY_test")
        stt._current_participant = participant

        stt._handle_message({"transcript": "", "confidence": None, "is_final": True})
        stt._handle_message({"transcript": "   ", "confidence": None, "is_final": True})

        assert await stt.output.collect(timeout=0) == []

    async def test_send_failure_does_not_propagate(
        self, stt_with_failing_socket, participant
    ):
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16), sample_rate=16000, format="s16"
        )
        await stt_with_failing_socket.process_audio(pcm, participant=participant)

    async def test_audio_before_start_is_dropped(self, participant):
        """process_audio must not block forever when start() never ran."""
        stt = STT(api_key="KEY_test")
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16), sample_rate=16000, format="s16"
        )

        await asyncio.wait_for(
            stt.process_audio(pcm, participant=participant), timeout=1.0
        )

    async def test_integer_confidence_is_kept(self, participant):
        stt = STT(api_key="KEY_test")
        stt._current_participant = participant

        stt._handle_message({"transcript": "hello", "confidence": 1, "is_final": True})
        items = await stt.output.collect(timeout=0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert transcripts[0].confidence == 1.0

    async def test_error_payload_emits_no_transcript(self, participant):
        stt = STT(api_key="KEY_test")
        stt._current_participant = participant

        stt._handle_message(
            {
                "errors": [
                    {
                        "code": "40001",
                        "title": "Invalid Parameter",
                        "detail": "Unsupported input_format 'zzz'.",
                    }
                ]
            }
        )

        assert await stt.output.collect(timeout=0) == []


@pytest.mark.skipif(not os.getenv("TELNYX_API_KEY"), reason="TELNYX_API_KEY not set")
@pytest.mark.integration
class TestTelnyxSTTIntegration:
    """Integration tests against the real Telnyx streaming STT."""

    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    @pytest.fixture
    async def telnyx_stt(self):
        stt = STT()
        await stt.start()
        yield stt
        await stt.close()

    @pytest.fixture
    async def telnyx_stt_8khz(self):
        stt = STT(sample_rate=8000)
        await stt.start()
        yield stt
        await stt.close()

    async def test_transcribe_mia_audio_16khz(
        self, telnyx_stt, mia_audio_16khz, participant
    ):
        await telnyx_stt.process_audio(mia_audio_16khz, participant=participant)

        items = await telnyx_stt.output.collect(timeout=15.0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert transcripts, "No Transcript emitted by Telnyx STT"
        finals = [t for t in transcripts if t.final]
        assert finals, "No final Transcript emitted by Telnyx STT"
        assert "forgotten treasures" in " ".join(t.text for t in finals).lower()
        assert transcripts[0].participant == participant

    async def test_transcribe_at_telephony_sample_rate(
        self, telnyx_stt_8khz, mia_audio_16khz, participant
    ):
        """8 kHz is what TelnyxMediaStream produces from PCMU telephony audio."""
        await telnyx_stt_8khz.process_audio(mia_audio_16khz, participant=participant)

        items = await telnyx_stt_8khz.output.collect(timeout=15.0)

        finals = [i for i in items if isinstance(i, Transcript) and i.final]
        assert finals, "No final Transcript emitted at 8kHz"
        assert "forgotten treasures" in " ".join(t.text for t in finals).lower()

    async def test_interim_results_stream_partial_transcripts(
        self, mia_audio_16khz, participant
    ):
        """Covers the replacement-mode path, which finals-only engines never hit.

        ``interim_results`` is honoured per engine: Speechmatics streams
        partials, while the default Telnyx engine returns finals only.
        """
        stt = STT(transcription_engine="Speechmatics", interim_results=True)
        await stt.start()
        try:
            await stt.process_audio(mia_audio_16khz, participant=participant)
            items = await stt.output.collect(timeout=15.0)
        finally:
            await stt.close()

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert [t for t in transcripts if not t.final], "No interim Transcript emitted"
        assert [t for t in transcripts if t.final], "No final Transcript emitted"
