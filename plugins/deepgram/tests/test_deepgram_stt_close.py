from vision_agents.plugins import deepgram


class TestDeepgramSTTClose:
    async def test_close_closes_http_client(self):
        stt = deepgram.STT(api_key="fake")
        httpx_client = stt.client._client_wrapper.httpx_client.httpx_client

        assert httpx_client.is_closed is False
        await stt.close()
        assert httpx_client.is_closed is True
