import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import inworld

# Load environment variables
load_dotenv()


class TestInworldTTS:
    @pytest.fixture
    async def tts(self) -> inworld.TTS:
        return inworld.TTS()

    @pytest.mark.skipif(
        os.getenv("INWORLD_API_KEY") is None, reason="INWORLD_API_KEY not set"
    )
    @pytest.mark.integration
    async def test_inworld_tts_convert_text_to_audio(self, tts: inworld.TTS):
        text = "Hello from Inworld AI."

        out = []
        async for item in tts.send_iter(text):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
