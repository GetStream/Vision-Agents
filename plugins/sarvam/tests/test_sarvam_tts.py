"""Tests for the Sarvam TTS plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.tts.testing import TTSSession
from vision_agents.plugins.sarvam import TTS

load_dotenv()


class TestSarvamTTS:
    """Unit tests for Sarvam TTS configuration."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            TTS()

    async def test_default_configuration(self):
        tts = TTS(api_key="sk_test")
        assert tts.model == "bulbul:v3"
        assert tts.language == "hi-IN"
        assert tts.speaker == "shubh"
        assert tts.sample_rate == 24000
        assert tts.provider_name == "sarvam"

    async def test_invalid_model_rejected(self):
        with pytest.raises(ValueError, match="Unsupported Sarvam TTS model"):
            TTS(api_key="sk_test", model="not-a-model")

    async def test_custom_speaker_and_language(self):
        tts = TTS(api_key="sk_test", language="en-IN", speaker="ritu")
        assert tts.language == "en-IN"
        assert tts.speaker == "ritu"

    async def test_v3_beta_model_accepted(self):
        tts = TTS(api_key="sk_test", model="bulbul:v3-beta", speaker="shubh")
        assert tts.model == "bulbul:v3-beta"

    async def test_incompatible_speaker_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            TTS(api_key="sk_test", model="bulbul:v2", speaker="shubh")

    async def test_v3_speaker_on_v2_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            TTS(api_key="sk_test", model="bulbul:v3", speaker="hitesh")


@pytest.mark.skipif(not os.getenv("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set")
@pytest.mark.integration
class TestSarvamTTSIntegration:
    """Integration tests against the real Sarvam streaming TTS."""

    @pytest.fixture
    async def tts(self):
        t = TTS(language="en-IN", speaker="shubh")
        try:
            yield t
        finally:
            await t.close()

    async def test_stream_audio_yields_chunks(self, tts):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)

        await tts.send("Hello from Sarvam.")
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0
