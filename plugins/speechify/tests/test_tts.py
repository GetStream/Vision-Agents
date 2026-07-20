import os

import pytest
from vision_agents.plugins import speechify


def _require_speechify_api_key() -> str:
    api_key = os.getenv("SPEECHIFY_API_KEY")
    if not api_key:
        pytest.fail(
            "Speechify integration tests require SPEECHIFY_API_KEY. "
            "Set SPEECHIFY_API_KEY in the environment or in a .env file before "
            "running tests marked with @pytest.mark.integration.",
            pytrace=False,
        )
    return api_key


class TestSpeechifyTTS:
    def test_defaults(self) -> None:
        tts = speechify.TTS(api_key="fake")
        assert tts.model == "simba-3.2"
        assert tts.voice_id == "geffen_32"
        assert tts.language is None

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)
        with pytest.raises(ValueError):
            speechify.TTS()

    async def test_close_closes_http_client(self) -> None:
        tts = speechify.TTS(api_key="fake")
        httpx_client = tts.client._client_wrapper.httpx_client.httpx_client
        assert httpx_client.is_closed is False
        await tts.close()
        assert httpx_client.is_closed is True


@pytest.mark.integration
class TestSpeechifyTTSIntegration:
    @pytest.fixture
    async def tts(self) -> speechify.TTS:
        return speechify.TTS(api_key=_require_speechify_api_key())

    async def test_speechify_convert_text_to_audio(self, tts):
        out = []
        async for item in tts.send_iter("Hello from Speechify!"):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
