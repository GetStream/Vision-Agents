import pytest
import pytest_asyncio

from vision_agents.plugins import pocket
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession


class TestPocketTTS:
    @pytest_asyncio.fixture
    async def tts(self) -> pocket.TTS:
        tts_instance = pocket.TTS()
        await tts_instance.warmup()
        return tts_instance

    @pytest.mark.integration
    async def test_pocket_tts_convert_text_to_audio_manual_test(self, tts: pocket.TTS):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)

    @pytest.mark.integration
    async def test_pocket_tts_convert_text_to_audio(self, tts: pocket.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Pocket TTS."

        await tts.send(text)
        await session.wait_for_result(timeout=30.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_pocket_tts_with_custom_voice_path(self):
        tts_instance = pocket.TTS(
            voice="hf://kyutai/tts-voices/alba-mackenna/casual.wav"
        )
        await tts_instance.warmup()

        tts_instance.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts_instance)
        text = "Testing with a custom voice path."

        await tts_instance.send(text)
        await session.wait_for_result(timeout=30.0)

        assert not session.errors
        assert len(session.speeches) > 0
