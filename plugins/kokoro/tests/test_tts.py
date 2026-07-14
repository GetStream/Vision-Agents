import numpy as np
import pytest
from vision_agents.plugins import kokoro

from conftest import skip_if_huggingface_model_unavailable


class FakePipeline:
    def __call__(self, *args, **kwargs):
        yield None, None, np.array([0.0, 0.25, -0.25], dtype=np.float32)


class TestKokoroTTS:
    async def test_kokoro_tts_uses_injected_pipeline(self):
        tts = kokoro.TTS(client=FakePipeline())
        try:
            out = [item async for item in tts.send_iter("Hello")]
        finally:
            await tts.close()

        assert tts.provider_name == "kokoro"
        assert out[0].data
        assert out[-1].final


@pytest.mark.integration
class TestKokoroIntegration:
    @pytest.fixture
    async def tts(self):
        tts_instance = kokoro.TTS()
        try:
            await tts_instance.warmup()
        except Exception as exc:
            await tts_instance.close()
            skip_if_huggingface_model_unavailable(exc, "Kokoro model")
            raise
        try:
            yield tts_instance
        finally:
            await tts_instance.close()

    async def test_kokoro_tts_convert_text_to_audio(self, tts):
        text = "Hello from Kokoro TTS."

        out = []
        async for item in tts.send_iter(text):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final
