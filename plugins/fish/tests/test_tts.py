from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import fish

# Load environment variables
load_dotenv()


def test_fish_tts_uses_s2_1_pro_by_default():
    tts = fish.TTS(client=MagicMock())

    assert tts.model == "s2.1-pro"


@pytest.mark.asyncio
async def test_fish_tts_forwards_s2_1_model_to_sdk():
    client = MagicMock()
    stream = object()
    client.tts.awaitable.return_value = stream
    tts = fish.TTS(client=client, model="s2.1-pro-free")

    with patch(
        "vision_agents.plugins.fish.tts.PcmData.from_response",
        return_value=MagicMock(),
    ):
        await tts.stream_audio("Hello from Fish Audio S2.1!")

    request = client.tts.awaitable.call_args.args[0]
    assert request.text == "Hello from Fish Audio S2.1!"
    client.tts.awaitable.assert_called_once_with(request, backend="s2.1-pro-free")


@pytest.mark.integration
class TestFishTTS:
    @pytest.fixture
    async def tts(self) -> fish.TTS:
        return fish.TTS()

    @pytest.fixture
    async def tts_free(self) -> fish.TTS:
        return fish.TTS(model="s2.1-pro-free")

    async def test_fish_tts_convert_text_to_audio(self, tts: fish.TTS):
        text = "Hello from Fish Audio S2.1! [laugh] This is amazing."

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

    async def test_fish_tts_s2_1_free_model(self, tts_free: fish.TTS):
        text = "Hello from Fish Audio S2.1 Free."

        out = []
        async for item in tts_free.send_iter(text):
            out.append(item)
        assert len(out) > 0
