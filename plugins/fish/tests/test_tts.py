from collections.abc import AsyncIterator
from typing import cast

import pytest
from dotenv import load_dotenv
from fish_audio_sdk import Session, TTSRequest
from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins import fish

# Load environment variables
load_dotenv()


class FakeTTSResource:
    def __init__(self) -> None:
        self.request: TTSRequest | None = None
        self.backend: fish.FishTTSModel | None = None

    def awaitable(
        self, request: TTSRequest, *, backend: fish.FishTTSModel
    ) -> AsyncIterator[bytes]:
        self.request = request
        self.backend = backend

        async def stream() -> AsyncIterator[bytes]:
            yield b"\x01\x00\x02\x00"

        return stream()


class FakeSession:
    def __init__(self) -> None:
        self.tts = FakeTTSResource()


@pytest.mark.parametrize(
    ("configured_model", "expected_model"),
    [(None, "s2.1-pro"), ("s2.1-pro-free", "s2.1-pro-free")],
)
async def test_fish_tts_streams_s2_1_audio(
    configured_model: fish.FishTTSModel | None,
    expected_model: fish.FishTTSModel,
):
    client = FakeSession()
    kwargs = {"model": configured_model} if configured_model else {}
    tts = fish.TTS(client=cast(Session, client), **kwargs)

    response = await tts.stream_audio("Hello from Fish Audio S2.1!")
    chunks = [chunk async for chunk in cast(AsyncIterator[PcmData], response)]

    assert tts.model == expected_model
    assert client.tts.backend == expected_model
    assert client.tts.request is not None
    assert client.tts.request.text == "Hello from Fish Audio S2.1!"
    assert len(chunks) == 1
    assert chunks[0].sample_rate == 16000
    assert chunks[0].channels == 1
    assert chunks[0].samples.tolist() == [1, 2]


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
