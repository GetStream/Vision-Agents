import asyncio
from typing import AsyncIterator, cast

from anam import AgentAudioInputConfig, AgentAudioInputStream, Session
import pytest
from getstream.video.rtc import audio_track
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import (
    RealtimeAudioOutputDoneEvent,
    RealtimeAudioOutputEvent,
)
from vision_agents.core.tts.events import TTSAudioEvent
from vision_agents.core.turn_detection import TurnStartedEvent
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.anam.anam_avatar_publisher import AnamAvatarPublisher


def _make_publisher(**overrides) -> AnamAvatarPublisher:
    default_kwargs = {
        "avatar_id": "test-avatar",
        "api_key": "test-key",
    }
    return AnamAvatarPublisher(**{**default_kwargs, **overrides})


class DummyAgent:
    def __init__(self):
        self.events = EventManager()
        self.events.register(TTSAudioEvent)
        self.events.register(RealtimeAudioOutputEvent)
        self.events.register(RealtimeAudioOutputDoneEvent)
        self.events.register(TurnStartedEvent)


class FakeAudioInputStream:
    def __init__(self):
        self.chunks: list[bytes] = []
        self.sequence_ended = False

    async def send_audio_chunk(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    async def end_sequence(self) -> None:
        self.sequence_ended = True


class FakeSession:
    def __init__(self):
        self.configs: list[AgentAudioInputConfig] = []
        self.stream = FakeAudioInputStream()

    def create_agent_audio_input_stream(
        self, config: AgentAudioInputConfig
    ) -> AgentAudioInputStream:
        self.configs.append(config)
        return cast(AgentAudioInputStream, self.stream)

    async def audio_frames(self) -> AsyncIterator[object]:
        while True:
            await asyncio.sleep(3600)
            yield None

    async def video_frames(self) -> AsyncIterator[object]:
        while True:
            await asyncio.sleep(3600)
            yield None


def _make_pcm() -> PcmData:
    return PcmData.from_bytes(
        b"\x01\x00" * 4000,
        sample_rate=16000,
        channels=1,
        format=AudioFormat.S16,
    )


class TestAnamAvatarPublisher:
    def test_init_with_all_args(self):
        pub = _make_publisher()
        assert pub.name == "anam_avatar"

    def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            _make_publisher(api_key=None)

    def test_init_missing_avatar_id_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANAM_AVATAR_ID", raising=False)
        with pytest.raises(ValueError, match="avatar ID"):
            _make_publisher(avatar_id=None)

    def test_init_custom_resolution(self):
        pub = _make_publisher(width=640, height=480)
        track = pub.publish_video_track()
        assert isinstance(track, QueuedVideoTrack)
        assert track.width == 640
        assert track.height == 480

    def test_init_odd_width_raises(self):
        with pytest.raises(ValueError, match="width must be a positive even integer"):
            _make_publisher(width=641, height=480)

    def test_init_odd_height_raises(self):
        with pytest.raises(ValueError, match="height must be a positive even integer"):
            _make_publisher(width=640, height=481)

    def test_publish_video_track(self):
        pub = _make_publisher()
        assert isinstance(pub.publish_video_track(), QueuedVideoTrack)

    def test_publish_audio_track(self):
        pub = _make_publisher()
        assert isinstance(pub.publish_audio_track(), audio_track.AudioStreamTrack)

    async def test_attach_agent_subscribes_to_events(self):
        pub = _make_publisher()
        agent = DummyAgent()

        pub.attach_agent(agent)

        assert pub._real_agent is agent
        assert agent.events.has_subscribers(TTSAudioEvent)
        assert agent.events.has_subscribers(RealtimeAudioOutputEvent)
        assert agent.events.has_subscribers(RealtimeAudioOutputDoneEvent)
        assert agent.events.has_subscribers(TurnStartedEvent)

    async def test_creates_audio_input_stream_with_configured_sample_rate(self):
        pub = _make_publisher(audio_sample_rate=24000)
        session = FakeSession()
        pub._real_session = cast(Session, session)

        pub._create_audio_input_stream()

        assert pub._audio_input_stream is session.stream
        assert len(session.configs) == 1
        assert session.configs[0].sample_rate == 24000
        assert session.configs[0].channels == 1
        assert session.configs[0].encoding == "pcm_s16le"

    async def test_send_audio_sends_single_original_pcm_payload(self):
        pub = _make_publisher()
        session = FakeSession()
        pub._real_session = cast(Session, session)
        pcm = _make_pcm()

        await pub._send_audio(pcm)

        assert session.stream.chunks == [pcm.to_bytes()]
        assert len(session.configs) == 1

    async def test_connect_creates_audio_input_stream_before_receiver_tasks(self):
        pub = _make_publisher(audio_sample_rate=22050)
        session = FakeSession()
        pub._real_session = cast(Session, session)
        pub._connected.set()
        pub._session_ready.set()

        await pub._connect()

        assert pub._audio_input_stream is session.stream
        assert len(session.configs) == 1
        assert session.configs[0].sample_rate == 22050
        assert pub._audio_receiver_task is not None
        assert pub._video_receiver_task is not None

        pub._audio_receiver_task.cancel()
        pub._video_receiver_task.cancel()
        await asyncio.gather(
            pub._audio_receiver_task,
            pub._video_receiver_task,
            return_exceptions=True,
        )
