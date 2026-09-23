"""Tests for the Smallest AI TTS plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins.smallest import TTS

load_dotenv()


class TestSmallestTTS:
    """Unit tests for Smallest AI TTS configuration."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SMALLEST_API_KEY"):
            TTS()

    async def test_default_configuration(self):
        tts = TTS(api_key="sk_test")
        assert tts.model == "lightning_v3.1"
        assert tts.voice_id == "magnus"
        assert tts.sample_rate == 24000
        assert tts.provider_name == "smallest"

    async def test_invalid_model_rejected(self):
        with pytest.raises(ValueError, match="Unsupported Smallest AI TTS model"):
            TTS(api_key="sk_test", model="not-a-model")

    async def test_custom_voice_and_model(self):
        tts = TTS(api_key="sk_test", voice_id="meher", model="lightning_v3.1_pro")
        assert tts.voice_id == "meher"
        assert tts.model == "lightning_v3.1_pro"


@pytest.mark.skipif(
    not os.getenv("SMALLEST_API_KEY"), reason="SMALLEST_API_KEY not set"
)
@pytest.mark.integration
class TestSmallestTTSIntegration:
    """Integration tests against the real Smallest AI streaming TTS."""

    @pytest.fixture
    async def tts(self):
        t = TTS(voice_id="magnus")
        try:
            yield t
        finally:
            await t.close()

    async def test_stream_audio_yields_chunks(self, tts):
        out = []
        async for item in tts.send_iter(
            "This is a test of the Smallest AI text-to-speech API."
        ):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
