import os

import pytest
from vision_agents.plugins import cartesia


def _require_cartesia_api_key() -> str:
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        pytest.fail(
            "Cartesia integration tests require CARTESIA_API_KEY. "
            "Set CARTESIA_API_KEY in the environment or in a .env file before "
            "running tests marked with @pytest.mark.integration.",
            pytrace=False,
        )
    return api_key


class TestCartesiaTTS:
    def test_defaults_to_sonic_35(self) -> None:
        tts = cartesia.TTS(api_key="fake")
        assert tts.model_id == "sonic-3.5"


@pytest.mark.integration
class TestCartesiaTTSIntegration:
    @pytest.fixture
    async def tts(self) -> cartesia.TTS:
        return cartesia.TTS(api_key=_require_cartesia_api_key())

    async def test_cartesia_convert_text_to_audio(self, tts):
        out = []
        async for item in tts.send_iter("Hello from Cartesia!"):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
