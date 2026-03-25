"""Tests for LocalEdge and LocalConnection."""

import asyncio

import numpy as np
from aiortc import VideoStreamTrack
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.events import AudioReceivedEvent
from vision_agents.core.edge.types import Participant, User
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.plugins.local.devices import CameraDevice
from vision_agents.plugins.local.edge import LocalCall, LocalConnection, LocalEdge

from .conftest import _FakeAgent, _FakeAudioInput, _FakeAudioOutput, _make_transport


class TestLocalEdge:
    """Tests for LocalEdge class."""

    async def test_transport_with_custom_devices(self) -> None:
        fake_input = _FakeAudioInput(sample_rate=16000, channels=1)
        fake_output = _FakeAudioOutput(sample_rate=16000, channels=2)

        transport = LocalEdge(
            audio_input=fake_input,
            audio_output=fake_output,
        )

        assert transport._audio_input is fake_input
        assert transport._audio_output is fake_output
        assert transport._audio_input.sample_rate == 16000

    async def test_create_audio_track(self) -> None:
        transport = _make_transport()
        track = transport.create_audio_track()

        assert track is not None
        assert track._audio_output is transport._audio_output

    async def test_join_starts_microphone(self) -> None:
        fake_input = _FakeAudioInput(sample_rate=48000)
        transport = _make_transport(audio_input=fake_input)

        connection = await transport.join(_FakeAgent())

        assert transport._mic_task is not None
        assert connection is not None
        assert fake_input.started

        await transport.close()

    async def test_close_stops_audio(self) -> None:
        fake_input = _FakeAudioInput(sample_rate=48000)
        transport = _make_transport(audio_input=fake_input)

        await transport.join(_FakeAgent())
        await transport.close()

        assert transport._mic_task is None
        assert fake_input.stopped

    async def test_publish_tracks_starts_output(self) -> None:
        transport = _make_transport()
        track = transport.create_audio_track()

        await transport.publish_tracks(track, None)

        assert track._running
        assert track._playback_task is not None

        track.stop()

    async def test_authenticate_is_noop(self) -> None:
        transport = _make_transport()
        user = User(id="test", name="Test User")
        await transport.authenticate(user)

    async def test_add_track_subscriber_returns_none_for_unknown(self) -> None:
        transport = _make_transport()
        result = transport.add_track_subscriber("some-track-id")
        assert result is None

    async def test_add_track_subscriber_returns_video_track(self) -> None:
        transport = _make_transport(
            video_input=CameraDevice(index=0, name="Test Camera", device="0")
        )
        transport.create_video_track()

        result = transport.add_track_subscriber("local-video-track")

        assert result is not None
        assert result._device == "0"

    async def test_create_conversation_returns_in_memory(self) -> None:
        transport = _make_transport()
        user = User(id="test", name="Test")

        result = await transport.create_conversation(
            LocalCall(id="test"), user, "instructions"
        )
        assert isinstance(result, InMemoryConversation)
        assert result.instructions == "instructions"
        assert result.messages == []


class TestLocalConnection:
    """Tests for LocalConnection class."""

    async def test_idle_since_returns_zero(self) -> None:
        transport = _make_transport()
        connection = LocalConnection(transport)
        assert connection.idle_since() == 0.0

    async def test_wait_for_participant_returns_immediately(self) -> None:
        transport = _make_transport()
        connection = LocalConnection(transport)

        await asyncio.wait_for(
            connection.wait_for_participant(timeout=10.0), timeout=1.0
        )

    async def test_connection_close(self) -> None:
        fake_input = _FakeAudioInput(sample_rate=48000)
        transport = _make_transport(audio_input=fake_input)
        connection = await transport.join(_FakeAgent())

        await connection.close()
        assert transport._mic_task is None


class TestAudioReceivedEvent:
    """Tests for audio event emission."""

    async def test_emit_audio_event(self) -> None:
        transport = _make_transport()

        received_events: list[AudioReceivedEvent] = []

        @transport.events.subscribe
        async def on_audio(event: AudioReceivedEvent) -> None:
            received_events.append(event)

        data = np.array([[100], [200], [300], [400]], dtype=np.int16)
        transport._emit_audio_event(data)

        await transport.events.wait(timeout=2.0)

        assert len(received_events) == 1
        event = received_events[0]
        assert event.pcm_data is not None
        assert event.participant is not None
        assert isinstance(event.participant, Participant)
        assert event.participant.user_id == "local"
        assert event.participant.id == "local"

    async def test_mic_polling_with_custom_input(self) -> None:
        fake_input = _FakeAudioInput(sample_rate=16000, channels=1)
        transport = _make_transport(audio_input=fake_input)

        received_events: list[AudioReceivedEvent] = []
        got_event = asyncio.Event()

        @transport.events.subscribe
        async def on_audio(event: AudioReceivedEvent) -> None:
            received_events.append(event)
            got_event.set()

        await transport.join(_FakeAgent())

        fake_input.enqueue(np.array([100, 200, 300], dtype=np.int16))
        await asyncio.wait_for(got_event.wait(), timeout=2.0)

        assert len(received_events) >= 1
        event = received_events[0]
        assert event.pcm_data is not None
        assert event.pcm_data.sample_rate == 16000

        await transport.close()


class TestLocalEdgeVideo:
    """Tests for LocalEdge video functionality."""

    async def test_transport_with_video_input(self) -> None:
        transport = _make_transport(
            video_input=CameraDevice(index=0, name="Test Camera", device="0"),
            video_width=1280,
            video_height=720,
            video_fps=15,
        )

        assert transport._video_input == "0"
        assert transport._video_width == 1280
        assert transport._video_height == 720
        assert transport._video_fps == 15

    async def test_transport_without_video_input(self) -> None:
        transport = _make_transport()
        assert transport._video_input is None
        track = transport.create_video_track()
        assert track is None

    async def test_create_video_track_with_device(self) -> None:
        transport = _make_transport(
            video_input=CameraDevice(index=0, name="Test Camera", device="0")
        )
        track = transport.create_video_track()

        assert track is not None
        assert track._device == "0"

    async def test_publish_tracks_starts_video_forwarding(self) -> None:
        transport = _make_transport()
        video_track = VideoStreamTrack()

        await transport.publish_tracks(None, video_track)

        assert transport._video_forward_task is not None
        await cancel_and_wait(transport._video_forward_task)

    async def test_open_demo_for_agent_exists(self) -> None:
        transport = _make_transport()
        assert hasattr(transport, "open_demo_for_agent")

    async def test_output_video_track_is_always_available(self) -> None:
        transport = _make_transport()
        assert transport._output_video_track is not None

    async def test_close_cleans_up_video_display(self) -> None:
        transport = _make_transport()
        await transport.join(_FakeAgent())

        stopped = False

        class _FakeDisplay:
            async def stop(self) -> None:
                nonlocal stopped
                stopped = True

        transport._video_display = _FakeDisplay()  # type: ignore[assignment]
        await transport.close()

        assert stopped
        assert transport._video_display is None
