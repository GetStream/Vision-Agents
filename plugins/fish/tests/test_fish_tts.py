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

    @pytest_asyncio.fixture
    async def tts_legacy(self) -> fish.TTS:
        return fish.TTS(model="speech-1.5")

    @pytest.mark.integration
    async def test_fish_tts_convert_text_to_audio_manual_test(self, tts: fish.TTS):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)

    @pytest.mark.integration
    async def test_fish_tts_convert_text_to_audio(self, tts: fish.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Fish Audio S2! [laugh] This is amazing."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_fish_tts_s2_prosody_control(self, tts: fish.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "[whisper] This is a secret. [super happy] But this is great news!"

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_fish_tts_legacy_model(self, tts_legacy: fish.TTS):
        tts_legacy.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts_legacy)
        text = "Hello from Fish Audio legacy model."

        await tts_legacy.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0
