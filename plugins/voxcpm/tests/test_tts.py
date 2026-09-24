import asyncio
import base64
import io
import json
import os
import socket
import wave
from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from vision_agents.plugins import voxcpm
from vision_agents.plugins.voxcpm.tts import _WavStreamParser


def _wav_bytes(frame_count: int = 480) -> tuple[bytes, bytes]:
    pcm = b"\x01\x00\xff\xff" * (frame_count // 2)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48_000)
        wav_file.writeframes(pcm)
    return output.getvalue(), pcm


def _sse_event(event_type: str, **payload: object) -> bytes:
    event = {"type": event_type, **payload}
    return f"data: {json.dumps(event)}\n\n".encode()


@pytest.fixture
async def modelbest_server() -> AsyncIterator[tuple[str, dict[str, object]]]:
    state: dict[str, object] = {
        "events": [],
        "status": 200,
        "release": None,
    }

    async def speech(request: web.Request) -> web.StreamResponse:
        state["payload"] = await request.json()
        state["authorization"] = request.headers.get("Authorization")
        status = int(state["status"])
        if status != 200:
            raise web.HTTPUnauthorized(text="invalid key")

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in state["events"]:
            await response.write(event)
        release = state["release"]
        if isinstance(release, asyncio.Event):
            await release.wait()
        try:
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    runner = web.AppRunner(app)
    await runner.setup()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    site = web.SockSite(runner, listener)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}/v1", state
    finally:
        release = state["release"]
        if isinstance(release, asyncio.Event):
            release.set()
        await runner.cleanup()


class TestWavStreamParser:
    def test_parses_header_and_frames_split_across_chunks(self):
        wav_audio, expected_pcm = _wav_bytes()
        parser = _WavStreamParser()

        output = [
            parser.feed(wav_audio[:7]),
            parser.feed(wav_audio[7:43]),
            parser.feed(wav_audio[43:177]),
            parser.feed(wav_audio[177:]),
        ]
        parser.finish()

        assert b"".join(output) == expected_pcm
        assert parser.sample_rate == 48_000
        assert parser.channels == 1

    def test_rejects_non_wav_data(self):
        parser = _WavStreamParser()

        with pytest.raises(voxcpm.VoxCPMTTSError, match="RIFF/WAVE"):
            parser.feed(b"not a wave file")


class TestVoxCPMTTS:
    def test_requires_credentials_and_model(self, monkeypatch):
        monkeypatch.delenv("MODELBEST_API_KEY", raising=False)
        monkeypatch.delenv("MODELBEST_VOXCPM_MODEL_ID", raising=False)

        with pytest.raises(ValueError, match="MODELBEST_API_KEY"):
            voxcpm.TTS(model="model")
        with pytest.raises(ValueError, match="MODELBEST_VOXCPM_MODEL_ID"):
            voxcpm.TTS(api_key="key")

    def test_requires_prompt_audio_and_text_together(self):
        wav_audio, _ = _wav_bytes()

        with pytest.raises(ValueError, match="provided together"):
            voxcpm.TTS(
                api_key="key",
                model="model",
                prompt_audio=wav_audio,
            )

    async def test_streams_pcm_and_sends_clone_inputs(self, modelbest_server):
        base_url, state = modelbest_server
        wav_audio, expected_pcm = _wav_bytes()
        reference_audio, _ = _wav_bytes(160)
        slices = (wav_audio[:17], wav_audio[17:101], wav_audio[101:])
        state["events"] = [
            *[
                _sse_event(
                    "speech.audio.delta",
                    audio=base64.b64encode(fragment).decode(),
                )
                for fragment in slices
            ],
            _sse_event("speech.audio.done"),
        ]

        instance = voxcpm.TTS(
            api_key="test-key",
            model="voxcpm-model",
            base_url=base_url,
            ref_audio=reference_audio,
            prompt_audio=reference_audio,
            prompt_text="Reference transcript.",
        )
        try:
            output = [item async for item in instance.send_iter("Hello VoxCPM")]
        finally:
            await instance.close()

        pcm_chunks = [item.data for item in output if item.data is not None]
        assert b"".join(chunk.to_bytes() for chunk in pcm_chunks) == expected_pcm
        assert pcm_chunks[0].sample_rate == 48_000
        assert output[-1].final
        assert state["authorization"] == "Bearer test-key"
        payload = state["payload"]
        assert payload["model"] == "voxcpm-model"
        assert payload["input"] == "Hello VoxCPM"
        assert payload["voice"] == "default"
        assert payload["response_format"] == "wav"
        assert payload["stream"] is True
        assert payload["ref_audio"].startswith("data:audio/wav;base64,")
        assert payload["prompt_audio"].startswith("data:audio/wav;base64,")
        assert payload["prompt_text"] == "Reference transcript."

    async def test_missing_done_event_is_an_error(self, modelbest_server):
        base_url, state = modelbest_server
        wav_audio, _ = _wav_bytes()
        state["events"] = [
            _sse_event(
                "speech.audio.delta",
                audio=base64.b64encode(wav_audio).decode(),
            )
        ]
        instance = voxcpm.TTS(
            api_key="test-key", model="voxcpm-model", base_url=base_url
        )
        try:
            with pytest.raises(voxcpm.VoxCPMTTSError, match="speech.audio.done"):
                [item async for item in instance.send_iter("Hello")]
        finally:
            await instance.close()

    async def test_http_error_includes_status_and_message(self, modelbest_server):
        base_url, state = modelbest_server
        state["status"] = 401
        instance = voxcpm.TTS(
            api_key="invalid-key", model="voxcpm-model", base_url=base_url
        )
        try:
            with pytest.raises(voxcpm.VoxCPMTTSError, match="HTTP 401: invalid key"):
                [item async for item in instance.send_iter("Hello")]
        finally:
            await instance.close()

    async def test_sse_error_event_is_an_error(self, modelbest_server):
        base_url, state = modelbest_server
        state["events"] = [_sse_event("error", error="quota exceeded")]
        instance = voxcpm.TTS(
            api_key="test-key", model="voxcpm-model", base_url=base_url
        )
        try:
            with pytest.raises(voxcpm.VoxCPMTTSError, match="quota exceeded"):
                [item async for item in instance.send_iter("Hello")]
        finally:
            await instance.close()

    async def test_stop_audio_ends_an_in_flight_stream(self, modelbest_server):
        base_url, state = modelbest_server
        wav_audio, _ = _wav_bytes()
        state["events"] = [
            _sse_event(
                "speech.audio.delta",
                audio=base64.b64encode(wav_audio).decode(),
            )
        ]
        release_event = asyncio.Event()
        state["release"] = release_event
        instance = voxcpm.TTS(
            api_key="test-key", model="voxcpm-model", base_url=base_url
        )
        try:
            stream = await instance.stream_audio("A long utterance")
            first = await anext(stream)
            await instance.stop_audio()
            remaining = [chunk async for chunk in stream]

            release_event.set()
            state["release"] = None
            state["events"] = [
                _sse_event(
                    "speech.audio.delta",
                    audio=base64.b64encode(wav_audio).decode(),
                ),
                _sse_event("speech.audio.done"),
            ]
            follow_up = [item async for item in instance.send_iter("Hello again")]
        finally:
            release_event.set()
            await instance.close()

        assert first.duration_ms > 0
        assert remaining == []
        assert follow_up[0].data
        assert follow_up[-1].final


@pytest.mark.skipif(
    not os.getenv("MODELBEST_API_KEY") or not os.getenv("MODELBEST_VOXCPM_MODEL_ID"),
    reason="ModelBest VoxCPM credentials not set",
)
@pytest.mark.integration
class TestVoxCPMIntegration:
    async def test_streams_hosted_voxcpm_audio(self):
        instance = voxcpm.TTS()
        try:
            output = [item async for item in instance.send_iter("Hello from VoxCPM.")]
        finally:
            await instance.close()

        assert output[0].data
        assert output[-1].final
