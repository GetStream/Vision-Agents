import pytest
from dotenv import load_dotenv
from vision_agents.plugins import fish

# Load environment variables
load_dotenv()


@pytest.mark.integration
class TestFishTTS:
    @pytest.fixture
    async def tts(self) -> fish.TTS:
        return fish.TTS()

    @pytest.fixture
    async def tts_legacy(self) -> fish.TTS:
        return fish.TTS(model="speech-1.5")

    async def test_fish_tts_convert_text_to_audio(self, tts: fish.TTS):
        text = "Hello from Fish Audio S2! [laugh] This is amazing."

        out = []
        async for item in tts.send_iter(text):
            out.append(item)

        assert len(out) > 0

    async def test_fish_tts_s2_prosody_control(self, tts: fish.TTS):
        text = "[whisper] This is a secret. [super happy] But this is great news!"

        out = []
        async for item in tts.send_iter(text):
            out.append(item)
        assert len(out) > 0

    async def test_fish_tts_legacy_model(self, tts_legacy: fish.TTS):
        text = "Hello from Fish Audio legacy model."

        out = []
        async for item in tts_legacy.send_iter(text):
            out.append(item)
        assert len(out) > 0
