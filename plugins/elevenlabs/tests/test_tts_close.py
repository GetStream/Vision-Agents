from vision_agents.plugins import elevenlabs


class TestElevenLabsTTSClose:
    async def test_close_closes_http_client(self):
        tts = elevenlabs.TTS(api_key="fake")
        httpx_client = tts.client._client_wrapper.httpx_client.httpx_client

        assert httpx_client.is_closed is False
        await tts.close()
        assert httpx_client.is_closed is True
