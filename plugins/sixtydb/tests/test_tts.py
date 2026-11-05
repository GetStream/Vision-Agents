import asyncio
import base64
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from getstream.video.rtc.track_util import PcmData
from vision_agents.plugins.sixtydb import TTS

PCM = b"\x00\x00\x01\x00" * 240


def wav_audio(rate=24000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(PCM)
    return output.getvalue()


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.requests.append(
                (
                    self.headers.get("Authorization"),
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            time.sleep(self.server.delay)
            self.send_response(self.server.status)
            self.send_header("Content-Type", self.server.content_type)
            self.send_header("Location", "/redirected")
            self.send_header("Content-Length", str(len(self.server.body)))
            self.end_headers()
            try:
                self.wfile.write(self.server.body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.requests = []
    http.status = 200
    http.delay = 0
    http.content_type = "audio/pcm"
    http.body = PCM
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield http
    http.shutdown()
    http.server_close()
    thread.join()


def provider(server, **kwargs):
    instance = TTS(api_key="local-test-key", voice_id="workspace-voice", **kwargs)
    instance._endpoint = f"http://127.0.0.1:{server.server_port}/tts-synthesize"
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["pcm", "wav", "json", "ndjson", "wrapped", "ndjson-wav"]
)
async def test_real_http_audio_and_send_iter(server, kind):
    encoded = base64.b64encode(PCM).decode()
    if kind == "wav":
        server.body, server.content_type = wav_audio(), "audio/wav"
    elif kind == "json":
        server.body = json.dumps(
            {
                "success": True,
                "backendResponse": {"audio_base64": encoded, "sample_rate": 24000},
            }
        ).encode()
        server.content_type = "application/json"
    elif kind == "ndjson-wav":
        encoded = base64.b64encode(wav_audio()).decode()
        server.body = (
            json.dumps({"audioContent": encoded})
            + "\n"
            + json.dumps({"audioContent": encoded})
        ).encode()
        server.content_type = "application/x-ndjson"
    elif kind in {"ndjson", "wrapped"}:
        if kind == "wrapped":
            encoded = base64.b64encode(
                json.dumps({"result": {"audioContent": encoded}}).encode()
            ).decode()
        server.body = (
            json.dumps({"type": "meta"})
            + "\n"
            + json.dumps({"result": {"audioContent": encoded}})
        ).encode()
        server.content_type = "application/x-ndjson"
    instance = provider(server, model="workspace-model", speed=1.2)
    try:
        chunks = [chunk async for chunk in instance.send_iter("Hello")]
        assert len(chunks) == 1 and chunks[0].final
        assert isinstance(chunks[0].data, PcmData)
        assert chunks[0].data.sample_rate == 24000
        assert chunks[0].data.samples.tobytes() == PCM * (
            2 if kind == "ndjson-wav" else 1
        )
        authorization, payload = server.requests[0]
        assert authorization == "Bearer local-test-key"
        assert payload == {
            "text": "Hello",
            "voice_id": "workspace-voice",
            "speed": 1.2,
            "model_id": "workspace-model",
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000},
            "timestamp_type": "NONE",
        }
        assert instance.streaming is False
    finally:
        await instance.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,content_type",
    [
        (b"", "audio/pcm"),
        (b"x", "audio/pcm"),
        (b"ID3invalid", "audio/pcm"),
        (wav_audio(16000), "audio/wav"),
        (b'{"success":false}', "application/json"),
        (b'{"audio_base64":"invalid"}', "application/json"),
        (b'{"encoding":"mp3","audio_base64":"AAAA"}', "application/json"),
        (b'{"sample_rate":48000,"audio_base64":"AAAA"}', "application/json"),
        (b'{"audio_base64":42}', "application/json"),
        (b'{"result":null}', "application/json"),
        (b"[]", "application/json"),
        (b"{invalid", "application/x-ndjson"),
        (b'{"audioContent":"AAAA"}\n{"type":"error"}', "application/x-ndjson"),
        (b"<html>error</html>", "text/html"),
    ],
)
async def test_reject_invalid_service_output(server, body, content_type):
    server.body, server.content_type = body, content_type
    instance = provider(server)
    with pytest.raises((ValueError, TypeError, wave.Error)):
        _ = [chunk async for chunk in instance.send_iter("Hello")]
    await instance.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500, 302, 307])
async def test_http_errors_and_redirects(server, status):
    server.status = status
    instance = provider(server)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        await instance.stream_audio("Hello")
    assert len(server.requests) == 1
    await instance.close()


@pytest.mark.asyncio
async def test_interrupt_discards_output_then_next_utterance_works(server):
    server.delay = 0.15
    instance = provider(server)

    async def collect():
        return [chunk async for chunk in instance.send_iter("First")]

    pending = asyncio.create_task(collect())
    while not server.requests:
        await asyncio.sleep(0.005)
    await instance.interrupt()
    assert await pending == []
    server.delay = 0
    assert len([chunk async for chunk in instance.send_iter("Second")]) == 1
    await instance.close()


@pytest.mark.asyncio
async def test_timeout_and_response_limit(server, monkeypatch):
    instance = provider(server, timeout=0.03)
    server.delay = 0.1
    with pytest.raises((RuntimeError, TimeoutError)):
        await instance.stream_audio("Hello")
    server.delay = 0
    instance.timeout = 1
    monkeypatch.setattr("vision_agents.plugins.sixtydb.tts.MAX_RESPONSE_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds"):
        await instance.stream_audio("Hello")
    await instance.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": 0},
        {"speed": float("nan")},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"model": " "},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TTS(api_key="test", voice_id="test", **kwargs)


def test_environment_credentials(monkeypatch):
    monkeypatch.setenv("SIXTYDB_API_KEY", "test")
    monkeypatch.setenv("SIXTYDB_VOICE_ID", "voice")
    instance = TTS()
    assert instance.api_key == "test" and instance.voice_id == "voice"
    monkeypatch.delenv("SIXTYDB_API_KEY")
    with pytest.raises(ValueError, match="SIXTYDB_API_KEY"):
        TTS()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " ", "a" * 5001])
async def test_invalid_text_never_requests(server, text):
    instance = provider(server)
    with pytest.raises(ValueError, match="text"):
        await instance.stream_audio(text)
    assert server.requests == []
    await instance.close()
