import pytest
from vision_agents.plugins import cartesia


class TestCartesiaTTS:
    def test_defaults_to_sonic_35(self) -> None:
        tts = cartesia.TTS(api_key="fake")
        assert tts.model_id == "sonic-3.5"


@pytest.mark.integration
class TestCartesiaTTSIntegration:
    @pytest.fixture
    async def tts(self, cartesia_api_key_required) -> cartesia.TTS:
        return cartesia.TTS(api_key=cartesia_api_key_required)

    async def test_cartesia_convert_text_to_audio(self, tts):
        out = []
        async for item in tts.send_iter("Hello from Cartesia!"):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
