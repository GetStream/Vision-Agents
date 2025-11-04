import pytest
import pytest_asyncio
from dotenv import load_dotenv

from vision_agents.plugins import fish
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession

# Load environment variables
load_dotenv()


class TestFishTTS:
    @pytest_asyncio.fixture
    async def tts(self) -> fish.TTS:
        return fish.TTS()

    @pytest.mark.integration
    async def test_fish_tts_convert_text_to_audio_manual_test(self, tts: fish.TTS):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)

    @pytest.mark.integration
    async def test_fish_tts_convert_text_to_audio(self, tts: fish.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Fish Audio."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0
