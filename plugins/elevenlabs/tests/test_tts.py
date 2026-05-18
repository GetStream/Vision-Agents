import os

import pytest
from vision_agents.plugins import elevenlabs


@pytest.mark.skipif(
    os.getenv("ELEVENLABS_API_KEY") is None, reason="ELEVENLABS_API_KEY not set"
)
@pytest.mark.integration
class TestElevenLabsTTSIntegration:
    @pytest.fixture
    async def tts(self) -> elevenlabs.TTS:
        return elevenlabs.TTS()

    async def test_elevenlabs_with_real_api(self, tts):
        out = []
        async for item in tts.send_iter(
            "This is a test of the ElevenLabs text-to-speech API."
        ):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final


class TestElevenLabsTTS:
    async def test_close_closes_http_client(self):
        tts = elevenlabs.TTS(api_key="fake")
        httpx_client = tts.client._client_wrapper.httpx_client.httpx_client

        assert httpx_client.is_closed is False
        await tts.close()
        assert httpx_client.is_closed is True
