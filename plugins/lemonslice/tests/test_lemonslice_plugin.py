import asyncio
import json

import httpx
import numpy as np
import pytest
from getstream import AsyncStream
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.agents.inference import AudioOutputFlush, AudioOutputStream
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.lemonslice.lemonslice_avatar import LemonSliceAvatar
from vision_agents.plugins.lemonslice.track import AvatarInputTrack


def _make_avatar(**overrides) -> LemonSliceAvatar:
    default_kwargs = {
        "agent_id": "test-agent",
        "api_key": "lemonslice-key",
        "stream_api_key": "key",
        "stream_api_secret": "secret",
    }
    return LemonSliceAvatar(**{**default_kwargs, **overrides})


@pytest.fixture
def session_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def session_transport(session_requests: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        session_requests.append(request)
        return httpx.Response(200, json={"session_id": "session-1"})

    return httpx.MockTransport(handler)


@pytest.fixture
def call_events() -> list[dict]:
    return []


@pytest.fixture
def call_event_transport(call_events: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/event"):
            call_events.append(json.loads(request.content)["custom"])
        return httpx.Response(200, json={"duration": "0ms"})

    return httpx.MockTransport(handler)


class TestLemonSliceAvatar:
    async def test_init_with_agent_image_url_instead_of_id(self):
        avatar = _make_avatar(
            agent_id=None, agent_image_url="https://example.com/img.png"
        )
        assert avatar._client._agent_image_url == "https://example.com/img.png"

    async def test_init_missing_agent_identity_raises(self, monkeypatch):
        monkeypatch.delenv("LEMONSLICE_AGENT_ID", raising=False)
        with pytest.raises(ValueError, match="agent_id or agent_image_url"):
            _make_avatar(agent_id=None)

    async def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LEMONSLICE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            _make_avatar(api_key=None)

    async def test_init_missing_stream_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("STREAM_API_KEY", raising=False)
        monkeypatch.delenv("STREAM_API_SECRET", raising=False)
        with pytest.raises(ValueError, match="Stream API key and secret required"):
            _make_avatar(stream_api_key=None, stream_api_secret=None)

    async def test_video_output(self):
        avatar = _make_avatar(width=640, height=480)
        track = avatar.video_output()
        assert isinstance(track, QueuedVideoTrack)
        assert track.width == 640
        assert track.height == 480

    async def test_init_odd_width_raises(self):
        with pytest.raises(ValueError, match="width must be a positive even integer"):
            _make_avatar(width=641, height=480)

    async def test_init_odd_height_raises(self):
        with pytest.raises(ValueError, match="height must be a positive even integer"):
            _make_avatar(width=640, height=481)

    async def test_audio_output(self):
        avatar = _make_avatar()
        assert isinstance(avatar.audio_output(), AudioOutputStream)

    async def test_extra_params_are_sent_in_the_session_request(
        self,
        session_transport: httpx.MockTransport,
        session_requests: list[httpx.Request],
    ):
        avatar = _make_avatar(
            lemonslice_properties={"voice_id": "nova", "metadata": {"tier": "pro"}}
        )
        avatar._client._http_client = httpx.AsyncClient(transport=session_transport)

        await avatar._client.create_session(
            call_id="call-1", call_type="default", token="token", api_key="stream-key"
        )

        payload = json.loads(session_requests[0].content)
        assert payload["voice_id"] == "nova"
        assert payload["metadata"] == {"tier": "pro"}

    async def test_extra_params_do_not_override_transport_fields(
        self,
        session_transport: httpx.MockTransport,
        session_requests: list[httpx.Request],
    ):
        avatar = _make_avatar(
            lemonslice_properties={"transport_type": "websocket", "properties": {}}
        )
        avatar._client._http_client = httpx.AsyncClient(transport=session_transport)

        await avatar._client.create_session(
            call_id="call-1", call_type="default", token="token", api_key="stream-key"
        )

        payload = json.loads(session_requests[0].content)
        assert payload["transport_type"] == "stream"
        assert payload["properties"]["call_id"] == "call-1"

    async def test_end_utterance_and_interrupt_events_respect_audio_boundaries(
        self, call_events: list[dict], call_event_transport: httpx.MockTransport
    ):
        avatar = _make_avatar()
        manager = avatar._rtc_manager
        manager._client = AsyncStream(
            api_key="key", api_secret="secret", transport=call_event_transport
        )
        manager._call = manager._client.video.call("default", "call-1")
        manager._input_track = AvatarInputTrack(sample_rate=16000, channels=1)
        manager._connected = True

        for _ in range(3):
            await manager.send_audio(
                PcmData(
                    samples=np.zeros(480, dtype=np.int16),
                    sample_rate=24000,
                    format=AudioFormat.S16,
                    channels=1,
                )
            )

        await manager.flush()

        emitted_samples = 0
        while True:
            try:
                emitted_samples += (
                    await asyncio.wait_for(manager._input_track.recv(), timeout=0.05)
                ).samples
            except TimeoutError:
                break

        assert call_events[0] == {
            "type": "lemonslice.end_utterance",
            "pts": emitted_samples,
            "event_id": 1,
        }

        # An interruption discards the buffered audio, so announcing an
        # end-of-utterance PTS that covers it points the avatar at audio that
        # never arrives.
        await manager._input_track.write(
            PcmData(
                samples=np.zeros(16000, dtype=np.int16),
                sample_rate=16000,
                format=AudioFormat.S16,
                channels=1,
            )
        )

        stream = AudioOutputStream()
        avatar.attach_audio_input(stream)
        task = asyncio.create_task(avatar._process_audio_input())
        stream.send_nowait(AudioOutputFlush())
        await asyncio.sleep(0.05)
        await cancel_and_wait(task)

        assert [event["type"] for event in call_events] == [
            "lemonslice.end_utterance",
            "lemonslice.interrupt",
        ]
