import asyncio
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

    def test_invalid_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            audexum.TTS(api_key="fake", speed=5.0)

    async def test_stop_audio_cancels_request(self) -> None:
        release = asyncio.Event()

        async def handler(request: web.Request) -> web.Response:
            await release.wait()
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="fake", base_url=str(server.make_url("/v1")))
        try:
            pending = asyncio.ensure_future(tts.stream_audio("Hello"))
            await asyncio.sleep(0.1)
            await tts.stop_audio()
            pcm = await pending
            assert len(pcm.samples) == 0
        finally:
            release.set()
            await tts.close()
            await server.close()

    async def test_stop_audio_cancels_every_concurrent_request(self) -> None:
        release = asyncio.Event()
        arrived = 0

        async def handler(request: web.Request) -> web.Response:
            nonlocal arrived
            arrived += 1
            await release.wait()
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="fake", base_url=str(server.make_url("/v1")))
        try:
            first = asyncio.ensure_future(tts.stream_audio("First"))
            second = asyncio.ensure_future(tts.stream_audio("Second"))
            for _ in range(50):
                if arrived == 2:
                    break
                await asyncio.sleep(0.02)
            assert arrived == 2
            await tts.stop_audio()
            results = await asyncio.gather(first, second)
            assert [len(pcm.samples) for pcm in results] == [0, 0]
            assert not tts._pending
        finally:
            release.set()
            await tts.close()
            await server.close()

    async def test_request_after_stop_still_returns_audio(self) -> None:
        async def handler(request: web.Request) -> web.Response:
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="fake", base_url=str(server.make_url("/v1")))
        try:
            await tts.stop_audio()  # nothing in flight: a no-op
            pcm = await tts.stream_audio("Hello")
            assert len(pcm.samples) == 2400
        finally:
            await tts.close()
            await server.close()

    async def test_external_cancellation_is_not_swallowed(self) -> None:
        release = asyncio.Event()

        async def handler(request: web.Request) -> web.Response:
            await release.wait()
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="fake", base_url=str(server.make_url("/v1")))
        try:
            pending = asyncio.ensure_future(tts.stream_audio("Hello"))
            await asyncio.sleep(0.1)
            pending.cancel()  # a caller cancelling, not stop_audio()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            release.set()
            await tts.close()
            await server.close()

    async def test_close_with_pending_requests_returns_empty_audio(self) -> None:
        release = asyncio.Event()

        async def handler(request: web.Request) -> web.Response:
            await release.wait()
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="fake", base_url=str(server.make_url("/v1")))
        try:
            pending = [asyncio.ensure_future(tts.stream_audio(t)) for t in ("a", "b")]
            await asyncio.sleep(0.1)
            await tts.close()
            results = await asyncio.gather(*pending)
            assert [len(pcm.samples) for pcm in results] == [0, 0]
        finally:
            release.set()
            await server.close()

    async def test_caller_cancel_in_same_tick_as_stop_propagates(self) -> None:
        release = asyncio.Event()

        async def handler(request: web.Request) -> web.Response:
            await release.wait()
            return web.Response(body=_wav(), content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="fake", base_url=str(server.make_url("/v1")))
        try:
            pending = asyncio.ensure_future(tts.stream_audio("Hello"))
            await asyncio.sleep(0.1)
            await tts.stop_audio()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            release.set()
            await tts.close()
            await server.close()

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
        tts = audexum.TTS(api_key="sk_live_test", base_url=str(server.make_url("/v1")))
        try:
            pcm = await tts.stream_audio("Hello")
            assert seen["auth"] == "Bearer sk_live_test"
            assert seen["body"]["input"] == "Hello"
            assert seen["body"]["voice"] == "Nova"
            assert seen["body"]["speed"] == 1.0
            assert seen["body"]["response_format"] == "wav"
            assert pcm.sample_rate == 24000
            assert len(pcm.samples) == 2400
        finally:
            await tts.close()
            await server.close()

    @pytest.mark.parametrize("body", [b'{"error": "oops"}', _wav()[:20]])
    async def test_invalid_wav_raises_with_context(self, body: bytes) -> None:
        async def handler(request: web.Request) -> web.Response:
            return web.Response(body=body, content_type="audio/wav")

        app = web.Application()
        app.router.add_post("/v1/audio/speech", handler)
        server = TestServer(app)
        await server.start_server()
        tts = audexum.TTS(api_key="k", base_url=str(server.make_url("/v1")))
        try:
            with pytest.raises(RuntimeError, match="Audexum TTS returned invalid WAV"):
                await tts.stream_audio("Hello")
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
