import os
import pytest
import pytest_asyncio

from vision_agents.plugins import openai as openai_plugin
from vision_agents.core.tts.testing import TTSSession
from vision_agents.core.tts.manual_test import manual_tts_to_wav


class TestOpenAITTSIntegration:
    @pytest_asyncio.fixture
    async def tts(self) -> openai_plugin.TTS:  # type: ignore[name-defined]
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        return openai_plugin.TTS(api_key=api_key)

    @pytest.mark.integration
    async def test_openai_tts_speech(self, tts: openai_plugin.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)

        await tts.send("Hello from OpenAI TTS")

        result = await session.wait_for_result(timeout=20.0)
        assert not result.errors
        assert len(result.speeches) > 0

    @pytest.mark.integration
    async def test_openai_tts_manual_wav(self, tts: openai_plugin.TTS):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)
