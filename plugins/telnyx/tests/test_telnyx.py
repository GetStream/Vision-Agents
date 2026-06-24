"""Tests for the Telnyx plugin."""

import asyncio
import base64
import json
from datetime import datetime

from vision_agents.plugins import telnyx


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self.messages:
            raise RuntimeError("no more messages")
        return self.messages.pop(0)

    async def send_json(self, data):
        self.sent.append(data)


class DummyAudioTrack:
    def __init__(self):
        self.writes = []

    async def write(self, pcm):
        self.writes.append(pcm)


def test_import():
    assert telnyx.TelnyxCall is not None
    assert telnyx.TelnyxCallRegistry is not None
    assert telnyx.TelnyxMediaStream is not None


def test_create_call():
    webhook_data = {
        "data": {
            "payload": {
                "call_control_id": "v2:abc123",
                "from": "+1234567890",
                "to": "+10987654321",
                "state": "answered",
            }
        }
    }

    call = telnyx.TelnyxCall(
        call_control_id="v2:abc123",
        webhook_data=webhook_data,
    )

    assert call.call_control_id == "v2:abc123"
    assert call.from_number == "+1234567890"
    assert call.to_number == "+10987654321"
    assert call.call_status == "answered"
    assert call.ended_at is None


def test_end_call():
    call = telnyx.TelnyxCall(call_control_id="v2:abc123")
    assert call.ended_at is None

    call.end()
    assert call.ended_at is not None
    assert isinstance(call.ended_at, datetime)


def test_registry_create_get_remove():
    registry = telnyx.TelnyxCallRegistry()
    call = registry.create("v2:abc123")

    assert registry.get("v2:abc123") is call
    assert registry.validate("v2:abc123", call.token) is call

    removed = registry.remove("v2:abc123")
    assert removed is call
    assert removed.ended_at is not None
    assert registry.get("v2:abc123") is None


def test_registry_list_active():
    registry = telnyx.TelnyxCallRegistry()
    registry.create("v2:one")
    registry.create("v2:two")
    registry.remove("v2:two")

    active = registry.list_active()

    assert len(active) == 1
    assert active[0].call_control_id == "v2:one"


def test_media_stream_run_writes_audio():
    payload = base64.b64encode(bytes([0xFF] * 160)).decode("ascii")
    websocket = FakeWebSocket(
        [
            json.dumps({"event": "connected", "version": "1.0.0"}),
            json.dumps(
                {
                    "event": "start",
                    "stream_id": "stream-1",
                    "start": {
                        "call_control_id": "v2:abc123",
                        "media_format": {
                            "encoding": "PCMU",
                            "sample_rate": 8000,
                            "channels": 1,
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "event": "media",
                    "stream_id": "stream-1",
                    "media": {
                        "track": "inbound",
                        "payload": payload,
                    },
                }
            ),
            json.dumps({"event": "stop", "stream_id": "stream-1"}),
        ]
    )
    stream = telnyx.TelnyxMediaStream(websocket)
    dummy_track = DummyAudioTrack()
    stream.audio_track = dummy_track

    asyncio.run(stream.accept())
    asyncio.run(stream.run())

    assert websocket.accepted is True
    assert stream.stream_id == "stream-1"
    assert stream.call_control_id == "v2:abc123"
    assert stream.is_connected is False
    assert len(dummy_track.writes) == 1
    assert dummy_track.writes[0].sample_rate == 8000


def test_media_stream_send_audio():
    websocket = FakeWebSocket([])
    stream = telnyx.TelnyxMediaStream(websocket)
    pcm = telnyx.pcmu_to_pcm(bytes([0xFF] * 160))

    asyncio.run(stream.accept())
    asyncio.run(stream.send_audio(pcm))

    assert websocket.sent == []

    stream.stream_id = "stream-1"
    stream._started = True
    asyncio.run(stream.send_audio(pcm))

    assert websocket.sent[0]["event"] == "media"
    assert "payload" in websocket.sent[0]["media"]


def test_media_stream_start_recreates_audio_track_for_negotiated_format():
    websocket = FakeWebSocket(
        [
            json.dumps(
                {
                    "event": "start",
                    "stream_id": "stream-1",
                    "start": {
                        "call_control_id": "v2:abc123",
                        "media_format": {
                            "encoding": "L16",
                            "sample_rate": 16000,
                            "channels": 1,
                        },
                    },
                }
            ),
            json.dumps({"event": "stop", "stream_id": "stream-1"}),
        ]
    )
    stream = telnyx.TelnyxMediaStream(websocket)
    original_track = stream.audio_track

    asyncio.run(stream.accept())
    asyncio.run(stream.run())

    assert stream.media_format.encoding == "L16"
    assert stream.media_format.sample_rate == 16000
    assert stream.audio_track is not original_track


def test_media_stream_runs_cleanup_callbacks_on_close():
    cleanup_ran = False

    async def cleanup():
        nonlocal cleanup_ran
        cleanup_ran = True

    websocket = FakeWebSocket([json.dumps({"event": "stop", "stream_id": "stream-1"})])
    stream = telnyx.TelnyxMediaStream(websocket)
    stream.add_cleanup_callback(cleanup)

    asyncio.run(stream.accept())
    asyncio.run(stream.run())

    assert cleanup_ran
