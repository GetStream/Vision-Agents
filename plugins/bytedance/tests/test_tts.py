import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData

from vision_agents.plugins import bytedance

load_dotenv()


class TestBytedanceTTS:
    @pytest.fixture
    def tts(self) -> bytedance.TTS:
        return bytedance.TTS(api_key="test-key", speaker="my_voice", sample_rate=24000)

    def test_req_params_include_speaker_and_audio(self, tts):
        params = tts._req_params("hello")
        assert params["speaker"] == "my_voice"
        assert params["audio_params"] == {"format": "pcm", "sample_rate": 24000}
        assert params["text"] == "hello"

    def test_req_params_without_text(self, tts):
        params = tts._req_params()
        assert "text" not in params

    def test_req_params_includes_speech_rate_when_set(self):
        tts = bytedance.TTS(api_key="test-key", speech_rate=20)
        assert tts._req_params()["audio_params"]["speech_rate"] == 20

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BYTEDANCE_API_KEY", raising=False)
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("BYTEDANCE_APP_KEY", raising=False)
        monkeypatch.delenv("BYTEDANCE_ACCESS_KEY", raising=False)
        with pytest.raises(ValueError):
            bytedance.TTS()

    @pytest.mark.integration
    async def test_convert_text_to_audio(self):
        tts = bytedance.TTS()
        try:
            out = [chunk async for chunk in tts.send_iter("你好，世界。")]
        finally:
            await tts.close()

        assert len(out) > 0
        assert isinstance(out[0].data, PcmData)
        assert out[-1].final
