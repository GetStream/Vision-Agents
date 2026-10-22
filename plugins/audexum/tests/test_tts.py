import io
import os
import wave

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from vision_agents.plugins import audexum


def _wav(sample_rate: int = 24000, frames: int = 2400) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


class TestAudexumTTS:
    def test_defaults(self) -> None:
        tts = audexum.TTS(api_key="fake")
        assert tts.voice == "Nova"
        assert tts.speed == 1.0

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUDEXUM_API_KEY", raising=False)
        with pytest.raises(ValueError):
            audexum.TTS()

    async def test_request_shape_and_pcm_decoding(self) -> None:
        seen: dict = {}

        async def handler(request: web.Request) -> web.Response:
            seen["auth"] = request.headers["Authorization"]
            seen["body"] = await request.json()
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(
            api_key="sk_live_test", voice="Atlas", base_url=str(server.make_url("/v1"))
        )
        try:
            pcm = await tts.stream_audio("Hello")
            assert seen["auth"] == "Bearer sk_live_test"
            assert seen["body"]["input"] == "Hello"
            assert seen["body"]["voice"] == "Atlas"
            assert seen["body"]["response_format"] == "wav"
            assert pcm.sample_rate == 24000
            assert len(pcm.samples) == 2400
        finally:
            await tts.close()
            await server.close()

    async def test_http_error_raises(self) -> None:
        async def handler(request: web.Request) -> web.Response:
            return web.Response(status=402, text="out of credits")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="k", base_url=str(server.make_url("/v1")))
        try:
            with pytest.raises(RuntimeError, match="402"):
                await tts.stream_audio("Hello")
        finally:
            await tts.close()
            await server.close()


@pytest.mark.integration
class TestAudexumTTSIntegration:
    async def test_audexum_convert_text_to_audio(self) -> None:
        api_key = os.getenv("AUDEXUM_API_KEY")
        if not api_key:
            pytest.fail("Integration tests require AUDEXUM_API_KEY", pytrace=False)
        tts = audexum.TTS(api_key=api_key)
        try:
            out = [item async for item in tts.send_iter("Hello from Audexum!")]
        finally:
            await tts.close()
        assert len(out) > 0
        assert out[-1].final
