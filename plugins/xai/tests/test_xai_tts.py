import io
import os
import wave

import numpy as np
import pytest
from dotenv import load_dotenv

from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession
from vision_agents.plugins import xai
from vision_agents.plugins.xai.tts import VOICE_DESCRIPTIONS, XAITTS

load_dotenv()


class TestXAITTS:
    """Test suite for the xAI TTS class."""

    async def test_custom_voice_and_codec(self):
        t = XAITTS(
            api_key="k",
            voice="leo",
            codec="mp3",
            sample_rate=44100,
            bit_rate=192000,
        )
        assert t.voice == "leo"
        assert t.codec == "mp3"
        assert t.sample_rate == 44100
        assert t.bit_rate == 192000

    async def test_provider_name_is_xai(self):
        t = XAITTS(api_key="k")
        assert t.provider_name == "xai"

    async def test_payload_default(self):
        t = XAITTS(api_key="k")
        payload = t._build_payload("Hello")
        assert payload == {
            "text": "Hello",
            "voice_id": "eve",
            "language": "en",
            "output_format": {"codec": "pcm", "sample_rate": 24000},
        }

    async def test_payload_mp3_includes_bit_rate(self):
        t = XAITTS(api_key="k", codec="mp3", bit_rate=128000)
        payload = t._build_payload("Test")
        assert payload["output_format"]["bit_rate"] == 128000

    async def test_payload_pcm_excludes_bit_rate(self):
        t = XAITTS(api_key="k", codec="pcm", bit_rate=128000)
        payload = t._build_payload("Test")
        assert "bit_rate" not in payload["output_format"]

    async def test_payload_respects_language(self):
        t = XAITTS(api_key="k", language="pt-BR")
        assert t._build_payload("olá")["language"] == "pt-BR"

    async def test_decode_pcm_round_trip(self):
        t = XAITTS(api_key="k", codec="pcm", sample_rate=16000)
        samples = np.array([100, -200, 300, -400, 0], dtype=np.int16)
        pcm = t._decode_audio(samples.tobytes())
        assert pcm.sample_rate == 16000
        assert pcm.samples.tolist() == samples.tolist()

    async def test_decode_wav_round_trip(self):
        t = XAITTS(api_key="k", codec="wav", sample_rate=16000)
        samples = np.array([100, -200, 300, -400, 0], dtype=np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())
        pcm = t._decode_audio(buf.getvalue())
        assert pcm.sample_rate == 16000
        assert pcm.samples.tolist() == samples.tolist()

    async def test_decode_mulaw_silence(self):
        # In mu-law, both 0xFF and 0x7F decode to amplitude 0.
        t = XAITTS(api_key="k", codec="mulaw", sample_rate=8000)
        pcm = t._decode_audio(bytes([0xFF, 0x7F, 0xFF, 0x7F]))
        assert pcm.sample_rate == 8000
        assert pcm.samples.dtype == np.int16
        assert pcm.samples.tolist() == [0, 0, 0, 0]

    async def test_decode_mulaw_max_negative(self):
        # 0x00 is the maximum negative magnitude in mu-law (~0x00 = 0xFF).
        t = XAITTS(api_key="k", codec="mulaw", sample_rate=8000)
        pcm = t._decode_audio(bytes([0x00]))
        # Sign is set, exponent=7, mantissa=15 → magnitude is the largest
        # representable value before bias subtraction.
        assert int(pcm.samples[0]) < -30000

    async def test_decode_alaw_silence(self):
        # G.711 A-law: 0xD5 ^ 0x55 = 0x80 (sign set, exp=0, mantissa=0)
        # decodes to magnitude (0 << 4) | 0x08 = 8, with sign → -8.
        # 0x55 ^ 0x55 = 0x00 (sign clear, exp=0, mantissa=0) decodes to +8.
        # A-law has no exact zero — ±8 is the minimum representable magnitude.
        t = XAITTS(api_key="k", codec="alaw", sample_rate=8000)
        pcm = t._decode_audio(bytes([0xD5, 0x55]))
        assert pcm.sample_rate == 8000
        assert pcm.samples.dtype == np.int16
        assert pcm.samples.tolist() == [-8, 8]

    async def test_decode_alaw_max_magnitude(self):
        # G.711 A-law: 0x2A ^ 0x55 = 0x7F (sign clear, exp=7, mantissa=15)
        # decodes to ((15 << 4) | 0x108) << 6 = 0x7E00 = 32256.
        t = XAITTS(api_key="k", codec="alaw", sample_rate=8000)
        pcm = t._decode_audio(bytes([0x2A, 0xAA]))
        # 0xAA ^ 0x55 = 0xFF — same magnitude with sign set → -32256.
        assert pcm.samples.tolist() == [32256, -32256]

    async def test_decode_g711_static_dispatch(self):
        # _decode_g711 dispatches between mu-law and A-law decoders.
        mulaw = XAITTS._decode_g711(bytes([0xFF]), "mulaw")
        alaw = XAITTS._decode_g711(bytes([0x55]), "alaw")
        assert int(mulaw[0]) == 0
        assert int(alaw[0]) == 8

    async def test_voice_descriptions_complete(self):
        for voice in ("eve", "ara", "leo", "rex", "sal"):
            assert voice in VOICE_DESCRIPTIONS
            assert VOICE_DESCRIPTIONS[voice]

    async def test_exported_via_xai_namespace(self):
        # The TTS class is re-exported from the xai package.
        assert xai.TTS is XAITTS
        assert "eve" in xai.VOICE_DESCRIPTIONS

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_xai_tts_with_real_api(self):
        tts = xai.TTS(api_key=os.environ["XAI_API_KEY"])
        try:
            tts.set_output_format(sample_rate=16000, channels=1)
            session = TTSSession(tts)
            await tts.send("Hello from xAI text to speech.")
            result = await session.wait_for_result(timeout=30.0)
            assert not result.errors
            assert len(result.speeches) > 0
        finally:
            await tts.close()

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_xai_tts_manual_wav(self):
        tts = xai.TTS(api_key=os.environ["XAI_API_KEY"])
        try:
            path = await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
            assert path
        finally:
            await tts.close()
