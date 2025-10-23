from dotenv import load_dotenv
import os

import pytest

from vision_agents.plugins import cartesia
from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession

# Load environment variables
load_dotenv()


class TestCartesiaIntegration:
    def tts(self) -> cartesia.TTS:  # type: ignore[name-defined]
        api_key = os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            pytest.skip("CARTESIA_API_KEY env var not set – skipping live API test.")
        return cartesia.TTS(api_key=api_key)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cartesia_with_real_api(self):
        tts = self.tts()
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        await tts.send("Hello from Cartesia!")
        result = await session.wait_for_result(timeout=30)
        assert not result.errors
        assert len(result.speeches) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cartesia_tts_convert_text_to_audio_manual_test(self):
        api_key = os.environ.get("CARTESIA_API_KEY")
        if not api_key:
            pytest.skip(
                "CARTESIA_API_KEY env var not set – skipping manual playback test."
            )
        tts = self.tts()
        path = await manual_tts_to_wav(tts, sample_rate=16000, channels=1)
        print("Cartesia TTS audio written to:", path)
