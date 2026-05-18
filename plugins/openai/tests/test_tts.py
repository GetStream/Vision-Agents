import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.edge.types import Participant
from vision_agents.plugins import openai

load_dotenv()


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set"
)
@pytest.mark.integration
class TestOpenAITTS:
    @pytest.fixture
    async def tts(self) -> openai.TTS:
        return openai.TTS()

    async def test_openai_tts_speech(self, tts: openai.TTS):
        out = []
        async for item in tts.send_iter(
            "Hello from OpenAI TTS",
            participant=Participant(user_id="test", id="test", original=None),
        ):
            out.append(item)

        assert len(out) >= 1
        chunk = out[0]
        assert chunk.text == "Hello from OpenAI TTS"
        assert chunk.final
        assert chunk.data
