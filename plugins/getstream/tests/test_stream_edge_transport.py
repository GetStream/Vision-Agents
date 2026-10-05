import asyncio
from fractions import Fraction
from typing import AsyncIterator
from unittest.mock import AsyncMock
from uuid import uuid4

import aiortc
import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError
from getstream import AsyncStream
from getstream.models import QueryUsersPayload, UserRequest
from getstream.video import rtc
from getstream.video.async_call import Call as StreamCall
from getstream.video.rtc import (
    AudioFormat,
    AudioStreamTrack,
    ConnectionManager,
    PcmData,
)
from vision_agents.core import Agent
from vision_agents.core.edge.events import (
    AudioReceivedEvent,
    CallEndedEvent,
    ParticipantJoinedEvent,
    ParticipantLeftEvent,
    TrackAddedEvent,
    TrackRemovedEvent,
)
from vision_agents.core.edge.types import TrackType, User
from vision_agents.core.llm import LLM
from vision_agents.core.tts import TTS
from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge


class _SilentLLM(LLM):
    """LLM that never answers; the edge tests only need a constructible agent."""

    async def simple_response(self, text, participant=None):
        return
        yield


class _SilentTTS(TTS):
    """TTS without audio; it satisfies the agent's processing check."""

    async def stream_audio(self, text, *args, **kwargs):
        return b""

    async def stop_audio(self) -> None:
        pass


class _GreyVideoTrack(aiortc.VideoStreamTrack):
    """Video track that sends grey 640x480 frames at 30 fps."""

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = av.VideoFrame.from_ndarray(
            np.full((480, 640, 3), 128, dtype=np.uint8), format="rgb24"
        )
        frame.pts = pts
        frame.time_base = time_base
        return frame


@pytest.fixture
async def stream_edge(monkeypatch):
    monkeypatch.setenv("STREAM_API_KEY", "test-key")
    monkeypatch.setenv("STREAM_API_SECRET", "test-secret")
    return StreamEdge()


@pytest.fixture
async def edge() -> AsyncIterator[StreamEdge]:
    edge = StreamEdge()
    yield edge
    await edge.close()


@pytest.fixture
async def agent(edge: StreamEdge) -> Agent:
    agent = Agent(
        edge=edge,
        llm=_SilentLLM(),
        tts=_SilentTTS(),
        agent_user=User(id=f"agent-{uuid4()}", name="Agent"),
    )
    await edge.authenticate(agent.agent_user)
    return agent


@pytest.fixture
async def call(edge: StreamEdge, agent: Agent) -> StreamCall:
    return await edge.create_call(str(uuid4()))


@pytest.fixture
async def admin_client() -> AsyncIterator[AsyncStream]:
    """A REST client that the agent does not close when the call ends."""
    client = AsyncStream()
    yield client
    await client.aclose()


@pytest.fixture
async def peer(edge: StreamEdge, call: StreamCall) -> AsyncIterator[ConnectionManager]:
    """A second participant that is in the call before the agent joins."""
    user = User(id=f"peer-{uuid4()}", name="Peer")
    await edge.create_users([user])
    connection = await rtc.join(call, user.id)
    await connection.__aenter__()
    yield connection
    await connection.leave()


@pytest.fixture
async def peer_audio(peer: ConnectionManager) -> AsyncIterator[AudioStreamTrack]:
    """The second participant publishes a 440 Hz tone."""
    track = AudioStreamTrack(sample_rate=48000, channels=1)
    await peer.add_tracks(audio=track)
    tone = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 10000).astype(np.int16)

    async def write_tone() -> None:
        while True:
            await track.write(
                PcmData(
                    sample_rate=48000, format=AudioFormat.S16, samples=tone, channels=1
                )
            )
            await asyncio.sleep(0.1)

    task = asyncio.create_task(write_tone())
    yield track
    task.cancel()


@pytest.fixture
async def peer_video(peer: ConnectionManager) -> _GreyVideoTrack:
    """The second participant publishes grey video."""
    track = _GreyVideoTrack()
    await peer.add_tracks(video=track)
    return track


@pytest.fixture
def track_events(edge: StreamEdge) -> asyncio.Queue:
    queue: asyncio.Queue[TrackAddedEvent | TrackRemovedEvent] = asyncio.Queue()

    @edge.events.subscribe
    async def on_track(event: TrackAddedEvent | TrackRemovedEvent) -> None:
        queue.put_nowait(event)

    return queue


@pytest.fixture
def participant_events(edge: StreamEdge) -> asyncio.Queue:
    queue: asyncio.Queue[ParticipantJoinedEvent | ParticipantLeftEvent] = (
        asyncio.Queue()
    )

    @edge.events.subscribe
    async def on_participant(
        event: ParticipantJoinedEvent | ParticipantLeftEvent,
    ) -> None:
        queue.put_nowait(event)

    return queue


@pytest.fixture
def audio_events(edge: StreamEdge) -> asyncio.Queue:
    queue: asyncio.Queue[AudioReceivedEvent] = asyncio.Queue()

    @edge.events.subscribe
    async def on_audio(event: AudioReceivedEvent) -> None:
        queue.put_nowait(event)

    return queue


@pytest.fixture
def call_ended_events(edge: StreamEdge) -> asyncio.Queue:
    queue: asyncio.Queue[CallEndedEvent] = asyncio.Queue()

    @edge.events.subscribe
    async def on_call_ended(event: CallEndedEvent) -> None:
        queue.put_nowait(event)

    return queue


@pytest.mark.integration
class TestStreamConnection:
    async def test_not_idle_while_participant_in_call(
        self, edge: StreamEdge, agent: Agent, call: StreamCall, peer: ConnectionManager
    ):
        connection = await edge.join(agent, call)

        assert connection.idle_since() == 0.0

    async def test_idle_after_participant_leaves(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        participant_events: asyncio.Queue,
    ):
        connection = await edge.join(agent, call)

        await peer.leave()
        while not isinstance(
            await asyncio.wait_for(participant_events.get(), timeout=10),
            ParticipantLeftEvent,
        ):
            pass

        assert connection.idle_since() > 0

    async def test_wait_for_participant_returns_when_participant_in_call(
        self, edge: StreamEdge, agent: Agent, call: StreamCall, peer: ConnectionManager
    ):
        connection = await edge.join(agent, call)

        await connection.wait_for_participant(timeout=10)

    async def test_wait_for_participant_returns_when_participant_joins(
        self, edge: StreamEdge, agent: Agent, call: StreamCall
    ):
        connection = await edge.join(agent, call)
        waiting = asyncio.create_task(connection.wait_for_participant(timeout=10))
        user = User(id=f"peer-{uuid4()}", name="Peer")
        await edge.create_users([user])

        async with await rtc.join(call, user.id):
            await waiting

    async def test_agent_alone_does_not_end_wait_for_participant(
        self, edge: StreamEdge, agent: Agent, call: StreamCall
    ):
        connection = await edge.join(agent, call)

        with pytest.raises(asyncio.TimeoutError):
            await connection.wait_for_participant(timeout=2)


class TestStreamEdge:
    async def test_create_call_raises_before_authenticate(
        self, stream_edge: StreamEdge
    ):
        with pytest.raises(RuntimeError, match="not authenticated"):
            await stream_edge.create_call(call_id="call-1")

    async def test_close_releases_client_resources(self, stream_edge: StreamEdge):
        stream_edge._real_connection = AsyncMock()
        real_connection = stream_edge._real_connection

        assert stream_edge.client.client.is_closed is False
        assert stream_edge._real_connection is not None

        await stream_edge.close()

        assert stream_edge.client.client.is_closed is True
        assert stream_edge._real_connection is None
        real_connection.leave.assert_called_once()

    async def test_add_track_subscriber_for_unknown_track_gives_none(
        self, stream_edge: StreamEdge
    ):
        assert stream_edge.add_track_subscriber("unknown-session:2") is None

    @pytest.mark.integration
    async def test_authenticate_keeps_custom_fields_set_elsewhere(self) -> None:
        """Metadata is merged into the stored user instead of replacing it."""
        edge = StreamEdge()
        user_id = f"test-authenticate-{uuid4()}"
        try:
            await edge.client.upsert_users(
                UserRequest(
                    id=user_id, name="Stored name", custom={"set_elsewhere": "keep me"}
                )
            )

            await edge.authenticate(
                User(id=user_id, name="Agent", custom={"is_agent": True})
            )

            response = await edge.client.query_users(
                QueryUsersPayload(filter_conditions={"id": {"$eq": user_id}})
            )
            stored = response.data.users[0]
            assert stored.custom == {"set_elsewhere": "keep me", "is_agent": True}
            assert stored.name == "Agent"
        finally:
            await edge.close()

    @pytest.mark.integration
    async def test_authenticate_unsets_none_custom_fields(self) -> None:
        """None custom values remove stored fields without touching unrelated ones."""
        edge = StreamEdge()
        user_id = f"test-authenticate-unset-{uuid4()}"
        try:
            await edge.client.upsert_users(
                UserRequest(
                    id=user_id,
                    name="Stored name",
                    custom={
                        "set_elsewhere": "keep me",
                        "stt": {"provider": "deepgram"},
                    },
                )
            )

            await edge.authenticate(
                User(
                    id=user_id,
                    name="Agent",
                    custom={"is_agent": True, "stt": None},
                )
            )

            response = await edge.client.query_users(
                QueryUsersPayload(filter_conditions={"id": {"$eq": user_id}})
            )
            stored = response.data.users[0]
            assert stored.custom == {"set_elsewhere": "keep me", "is_agent": True}
            assert "stt" not in stored.custom
        finally:
            try:
                await edge.client.delete_users([user_id])
            finally:
                await edge.close()

    @pytest.mark.integration
    async def test_authenticate_omits_none_custom_on_create(self) -> None:
        """404 create path omits null-valued custom fields instead of storing them."""
        edge = StreamEdge()
        user_id = f"test-authenticate-create-none-{uuid4()}"
        try:
            await edge.authenticate(
                User(
                    id=user_id,
                    name="Agent",
                    custom={"is_agent": True, "stt": None},
                )
            )

            response = await edge.client.query_users(
                QueryUsersPayload(filter_conditions={"id": {"$eq": user_id}})
            )
            stored = response.data.users[0]
            assert stored.custom == {"is_agent": True}
            assert "stt" not in stored.custom
        finally:
            try:
                await edge.client.delete_users([user_id])
            finally:
                await edge.close()

    @pytest.mark.integration
    async def test_create_users_creates_users_that_do_not_exist_yet(self) -> None:
        edge = StreamEdge()
        user_id = f"test-create-users-{uuid4()}"
        try:
            created = await edge.create_users(
                [User(id=user_id, name="New user", custom={"is_agent": False})]
            )

            assert [u.id for u in created] == [user_id]
            assert created[0].name == "New user"
            assert created[0].custom == {"is_agent": False}
        finally:
            await edge.close()


@pytest.mark.integration
class TestStreamEdgeIntegration:
    async def test_participant_in_call_before_join_gives_participant_joined(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        participant_events: asyncio.Queue,
    ):
        await edge.join(agent, call)

        event = await asyncio.wait_for(participant_events.get(), timeout=10)
        assert isinstance(event, ParticipantJoinedEvent)
        assert event.participant.user_id == peer.user_id
        assert (
            event.participant.id
            == f"{peer.user_id}__{event.participant.original.session_id}"
        )

    async def test_published_audio_gives_track_added(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_audio: AudioStreamTrack,
        track_events: asyncio.Queue,
    ):
        await edge.join(agent, call)

        event = await asyncio.wait_for(track_events.get(), timeout=10)
        assert isinstance(event, TrackAddedEvent)
        assert event.track_type == TrackType.AUDIO
        assert event.participant is not None
        assert event.participant.user_id == peer.user_id

    async def test_participant_audio_gives_audio_received_with_joined_participant_id(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_audio: AudioStreamTrack,
        audio_events: asyncio.Queue,
        participant_events: asyncio.Queue,
    ):
        await edge.join(agent, call)

        joined = await asyncio.wait_for(participant_events.get(), timeout=10)
        while True:
            event = await asyncio.wait_for(audio_events.get(), timeout=10)
            if np.abs(event.pcm_data.samples).max() > 1000:
                break
        assert event.participant.user_id == peer.user_id
        assert event.participant.id == joined.participant.id

    async def test_agent_audio_reaches_participant(
        self, edge: StreamEdge, agent: Agent, call: StreamCall, peer: ConnectionManager
    ):
        received: asyncio.Queue[PcmData] = asyncio.Queue()
        peer.on("audio", received.put_nowait)
        await edge.join(agent, call)
        track = edge.create_audio_track()
        await edge.publish_tracks(track, None)
        tone = (np.sin(2 * np.pi * 440 * np.arange(48000) / 48000) * 10000).astype(
            np.int16
        )

        await track.write(
            PcmData(sample_rate=48000, format=AudioFormat.S16, samples=tone, channels=1)
        )

        while True:
            pcm = await asyncio.wait_for(received.get(), timeout=10)
            if np.abs(pcm.samples).max() > 1000:
                break
        assert pcm.participant.user_id == agent.agent_user.id

    async def test_republished_track_gives_track_added_again(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_audio: AudioStreamTrack,
        track_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        added = await asyncio.wait_for(track_events.get(), timeout=10)

        peer_audio.stop()
        removed = await asyncio.wait_for(track_events.get(), timeout=10)
        await peer.add_tracks(audio=AudioStreamTrack(sample_rate=48000, channels=1))
        added_again = await asyncio.wait_for(track_events.get(), timeout=10)

        assert [type(e) for e in (added, removed, added_again)] == [
            TrackAddedEvent,
            TrackRemovedEvent,
            TrackAddedEvent,
        ]
        assert added.track_id == removed.track_id == added_again.track_id

    async def test_participant_leave_gives_track_removed_and_participant_left(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_audio: AudioStreamTrack,
        track_events: asyncio.Queue,
        participant_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        added = await asyncio.wait_for(track_events.get(), timeout=10)
        await asyncio.wait_for(participant_events.get(), timeout=10)

        await peer.leave()

        removed = await asyncio.wait_for(track_events.get(), timeout=10)
        left = await asyncio.wait_for(participant_events.get(), timeout=10)
        assert isinstance(removed, TrackRemovedEvent)
        assert removed.track_id == added.track_id
        assert isinstance(left, ParticipantLeftEvent)
        assert left.participant.user_id == peer.user_id

    async def test_leave_after_unpublish_gives_one_track_removed(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_audio: AudioStreamTrack,
        track_events: asyncio.Queue,
        participant_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        await asyncio.wait_for(track_events.get(), timeout=10)
        await asyncio.wait_for(participant_events.get(), timeout=10)
        peer_audio.stop()
        removed = await asyncio.wait_for(track_events.get(), timeout=10)

        await peer.leave()

        left = await asyncio.wait_for(participant_events.get(), timeout=10)
        assert isinstance(removed, TrackRemovedEvent)
        assert isinstance(left, ParticipantLeftEvent)
        assert track_events.empty()

    async def test_track_added_after_participant_left_is_ignored(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_audio: AudioStreamTrack,
        track_events: asyncio.Queue,
        participant_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        added = await asyncio.wait_for(track_events.get(), timeout=10)
        track = edge._remote_tracks[added.track_id]
        await peer.leave()
        await asyncio.wait_for(track_events.get(), timeout=10)
        while not isinstance(
            await asyncio.wait_for(participant_events.get(), timeout=10),
            ParticipantLeftEvent,
        ):
            pass

        # The SDK sends tracks and SFU events on separate streams, so a
        # participant's track_added can arrive after its participant_left.
        edge._connection.emit("track_added", track)
        await asyncio.sleep(1)

        assert track_events.empty()
        assert edge.add_track_subscriber(added.track_id) is None

    async def test_call_end_gives_one_call_ended(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        admin_client: AsyncStream,
        call_ended_events: asyncio.Queue,
    ):
        await edge.join(agent, call)

        # The agent closes the edge client on call end, so another client ends the call.
        await admin_client.video.call(call.call_type, call.id).end()

        event = await asyncio.wait_for(call_ended_events.get(), timeout=10)
        assert event.call is call
        await asyncio.sleep(1)
        assert call_ended_events.empty()

    async def test_published_video_gives_yuv420p_frames(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer_video: _GreyVideoTrack,
        track_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        event = await asyncio.wait_for(track_events.get(), timeout=10)
        assert event.track_type == TrackType.VIDEO

        track = edge.add_track_subscriber(event.track_id)

        frame = await asyncio.wait_for(track.recv(), timeout=10)
        assert frame.format.name == "yuv420p"
        assert (frame.width, frame.height) == (640, 480)
        assert frame.time_base == Fraction(1, 90000)

    async def test_two_subscribers_both_get_frames(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer_video: _GreyVideoTrack,
        track_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        event = await asyncio.wait_for(track_events.get(), timeout=10)

        first = edge.add_track_subscriber(event.track_id)
        second = edge.add_track_subscriber(event.track_id)

        assert first is not second
        assert isinstance(
            await asyncio.wait_for(first.recv(), timeout=10), av.VideoFrame
        )
        assert isinstance(
            await asyncio.wait_for(second.recv(), timeout=10), av.VideoFrame
        )

    async def test_video_ends_when_publisher_leaves(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_video: _GreyVideoTrack,
        track_events: asyncio.Queue,
        participant_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        event = await asyncio.wait_for(track_events.get(), timeout=10)
        track = edge.add_track_subscriber(event.track_id)
        await asyncio.wait_for(track.recv(), timeout=10)

        await peer.leave()

        with pytest.raises(MediaStreamError):
            while True:
                await asyncio.wait_for(track.recv(), timeout=10)
        assert track.readyState == "ended"
        # StreamEdge drops the RemoteTrack before it sends ParticipantLeftEvent.
        while not isinstance(
            await asyncio.wait_for(participant_events.get(), timeout=10),
            ParticipantLeftEvent,
        ):
            pass
        assert edge.add_track_subscriber(event.track_id) is None

    async def test_video_ends_when_edge_closes(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer_video: _GreyVideoTrack,
        track_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        event = await asyncio.wait_for(track_events.get(), timeout=10)
        track = edge.add_track_subscriber(event.track_id)
        await asyncio.wait_for(track.recv(), timeout=10)

        await edge.close()

        with pytest.raises(MediaStreamError):
            while True:
                await asyncio.wait_for(track.recv(), timeout=10)

    @pytest.mark.xfail(
        strict=True,
        reason="stream-py: a video publish after stop() fails in the SDK "
        "(MediaError: no unused vp9 publish option, or RtcError: participant not found)",
    )
    async def test_republished_video_gives_frames(
        self,
        edge: StreamEdge,
        agent: Agent,
        call: StreamCall,
        peer: ConnectionManager,
        peer_video: _GreyVideoTrack,
        track_events: asyncio.Queue,
    ):
        await edge.join(agent, call)
        await asyncio.wait_for(track_events.get(), timeout=10)
        peer_video.stop()
        removed = await asyncio.wait_for(track_events.get(), timeout=10)
        assert isinstance(removed, TrackRemovedEvent)

        await peer.add_tracks(video=_GreyVideoTrack())

        added = await asyncio.wait_for(track_events.get(), timeout=10)
        assert isinstance(added, TrackAddedEvent)
        track = edge.add_track_subscriber(added.track_id)
        frame = await asyncio.wait_for(track.recv(), timeout=10)
        assert (frame.width, frame.height) == (640, 480)
