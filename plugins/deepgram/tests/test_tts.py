import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import deepgram

load_dotenv()


@pytest.mark.skipif(
    os.getenv("DEEPGRAM_API_KEY") is None, reason="DEEPGRAM_API_KEY not set"
)
@pytest.mark.integration
class TestDeepgramTTSIntegration:
    """Integration tests for Deepgram Flux TTS."""

    @pytest.fixture
    async def tts(self) -> deepgram.TTS:
        t = deepgram.TTS()
        yield t
        await t.close()

    async def test_deepgram_tts_convert_text_to_audio(self, tts: deepgram.TTS):
        out = [item async for item in tts.send_iter("Hello from Deepgram.")]

        assert len(out) > 0
        # First chunk must have some audio
        assert out[0].data
        # Last chunk must be marked as "final"
        assert out[-1].final

    async def test_connection_reused_across_calls(self, tts: deepgram.TTS):
        _ = [item async for item in tts.send_iter("Hello")]
        socket = tts._socket
        assert socket is not None

        _ = [item async for item in tts.send_iter("World")]
        assert tts._socket is socket

    async def test_speed_forwarded_produces_audio(self):
        tts = deepgram.TTS(speed=1.05)
        try:
            out = [item async for item in tts.send_iter("Hello there.")]
        finally:
            await tts.close()

        assert any(item.data for item in out)

    async def test_extended_sample_rate_produces_audio(self):
        tts = deepgram.TTS(sample_rate=24000)
        try:
            out = [item async for item in tts.send_iter("Hello there.")]
        finally:
            await tts.close()

        assert any(item.data for item in out)

    async def test_stop_audio_after_synthesis(self, tts: deepgram.TTS):
        out = [item async for item in tts.send_iter("Hello there.")]
        assert out

        await tts.stop_audio()
        assert tts._stop_event.is_set()


class TestDeepgramTTS:
    """Constructor validation that runs without a network connection."""

    def test_aura_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Flux"):
            deepgram.TTS(model="aura-2-thalia-en")

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample rate"):
            deepgram.TTS(sample_rate=12345)
