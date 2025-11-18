import pytest
from dotenv import load_dotenv

from vision_agents.plugins import inworld
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession

# Load environment variables
load_dotenv()


class TestInworldTTS:
    @pytest.fixture
    async def tts(self) -> inworld.TTS:
        return inworld.TTS()

    @pytest.mark.integration
    async def test_inworld_tts_convert_text_to_audio_manual_test(
        self, tts: inworld.TTS
    ):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)

    @pytest.mark.integration
    async def test_inworld_tts_convert_text_to_audio(self, tts: inworld.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Inworld AI."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0
