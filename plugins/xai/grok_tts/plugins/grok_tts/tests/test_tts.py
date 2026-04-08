import numpy as np
import pytest

from vision_agents.plugins.grok_tts.tts import TTS, VOICE_DESCRIPTIONS


class TestTTSInit:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="xAI API key is required"):
            TTS()

    def test_accepts_explicit_api_key(self):
        t = TTS(api_key="test-key")
        assert t._api_key == "test-key"
        assert t.voice == "eve"

    def test_reads_env_api_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        t = TTS()
        assert t._api_key == "env-key"

    def test_custom_voice(self):
        t = TTS(api_key="k", voice="leo")
        assert t.voice == "leo"

    def test_custom_codec_and_sample_rate(self):
        t = TTS(api_key="k", codec="mp3", sample_rate=44100, bit_rate=192000)
        assert t.codec == "mp3"
        assert t.sample_rate == 44100
        assert t.bit_rate == 192000


class TestPayload:
    def test_default_payload(self):
        t = TTS(api_key="k")
        payload = t._build_payload("Hello")
        assert payload == {
            "text": "Hello",
            "voice_id": "eve",
            "language": "en",
            "output_format": {"codec": "pcm", "sample_rate": 24000},
        }

    def test_mp3_payload_includes_bit_rate(self):
        t = TTS(api_key="k", codec="mp3", bit_rate=128000)
        payload = t._build_payload("Test")
        assert payload["output_format"]["bit_rate"] == 128000

    def test_pcm_payload_excludes_bit_rate(self):
        t = TTS(api_key="k", codec="pcm")
        payload = t._build_payload("Test")
        assert "bit_rate" not in payload["output_format"]


class TestDecodeAudio:
    def test_decode_pcm(self):
        t = TTS(api_key="k", codec="pcm", sample_rate=16000)
        samples = np.array([100, -200, 300], dtype=np.int16)
        pcm = t._decode_audio(samples.tobytes())
        assert pcm.sample_rate == 16000

    def test_decode_wav(self):
        import io
        import wave

        t = TTS(api_key="k", codec="wav", sample_rate=16000)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            samples = np.array([100, -200, 300], dtype=np.int16)
            wf.writeframes(samples.tobytes())
        pcm = t._decode_audio(buf.getvalue())
        assert pcm.sample_rate == 16000


class TestVoiceDescriptions:
    def test_all_voices_described(self):
        for voice in ("eve", "ara", "leo", "rex", "sal"):
            assert voice in VOICE_DESCRIPTIONS
