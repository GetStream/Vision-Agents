import pytest
from vision_agents.plugins import pocket


@pytest.mark.integration
class TestPocketTTS:
    @pytest.fixture
    async def tts(self) -> pocket.TTS:
        tts_instance = pocket.TTS()
        await tts_instance.warmup()
        return tts_instance

    @pytest.fixture
    async def tts_custom_voice(self) -> pocket.TTS:
        tts_instance = pocket.TTS(
            voice="hf://kyutai/tts-voices/alba-mackenna/casual.wav"
        )
        await tts_instance.warmup()
        return tts_instance

    async def test_pocket_tts_convert_text_to_audio(self, tts: pocket.TTS):
        text = "Hello from Pocket TTS."

        out = []
        async for item in tts.send_iter(text):
            out.append(item)

        assert len(out) > 0
        # Pocket returns a single output chunk
        assert out[0].data
        assert out[0].final

    async def test_pocket_tts_with_custom_voice_path(self, tts_custom_voice):
        text = "Testing with a custom voice path."
        out = []
        async for item in tts_custom_voice.send_iter(text):
            out.append(item)

        assert len(out) > 0
        # Pocket returns a single output chunk
        assert out[0].data
        assert out[0].final
