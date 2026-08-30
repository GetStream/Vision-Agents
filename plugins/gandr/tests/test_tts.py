import os

import pytest
from vision_agents.plugins import gandr


def _require_gandr_api_key() -> str:
    api_key = os.getenv("GANDR_API_KEY")
    if not api_key:
        pytest.skip("GANDR_API_KEY not set")
    return api_key


class TestGandrTTS:
    async def test_defaults(self) -> None:
        tts = gandr.TTS(api_key="fake")
        try:
            assert tts.model == "tts-1"
            assert tts.voice == "gandr-mia"
        finally:
            await tts.close()

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GANDR_API_KEY", raising=False)
        with pytest.raises(ValueError):
            gandr.TTS()

    async def test_input_over_character_cap_raises(self) -> None:
        tts = gandr.TTS(api_key="fake")
        try:
            with pytest.raises(ValueError):
                await tts.stream_audio("a" * 2001)
        finally:
            await tts.close()


@pytest.mark.integration
class TestGandrTTSIntegration:
    @pytest.fixture
    async def tts(self) -> gandr.TTS:
        return gandr.TTS(api_key=_require_gandr_api_key())

    async def test_gandr_convert_text_to_audio(self, tts):
        out = []
        async for item in tts.send_iter("Hello from Gandr!"):
            out.append(item)

        assert len(out) >= 1
        assert out[0].data
        assert out[-1].final
