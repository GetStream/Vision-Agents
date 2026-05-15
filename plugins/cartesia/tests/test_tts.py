import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import cartesia

# Load environment variables
load_dotenv()


@pytest.mark.skipif(
    os.getenv("CARTESIA_API_KEY") is None, reason="CARTESIA_API_KEY not set"
)
@pytest.mark.integration
class TestCartesiaTTSIntegration:
    @pytest.fixture
    async def tts(self) -> cartesia.TTS:
        return cartesia.TTS()

    async def test_cartesia_convert_text_to_audio(self, tts):
        out = []
        async for item in tts.send_iter("Hello from Cartesia!"):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
