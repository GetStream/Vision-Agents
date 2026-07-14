import asyncio
import time
from typing import Any, Optional
from uuid import uuid4

import aiortc
import numpy as np
import pytest
from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import Agent, User
from vision_agents.core.agents.inference import AudioOutputChunk, AudioOutputStream
from vision_agents.core.avatars import Avatar
from vision_agents.core.edge import Call, EdgeTransport
from vision_agents.core.events import EventManager
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors.base_processor import AudioPublisher
from vision_agents.core.stt import STT as BaseSTT
from vision_agents.core.tts import TTS
from vision_agents.core.turn_detection import TurnDetector
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable


class DummySTT(BaseSTT):
    turn_detection: bool = False

    async def process_audio(self, pcm_data, participant):
        pass


class DummySTTWithTurnDetection(BaseSTT):
    turn_detection: bool = True

    async def process_audio(self, pcm_data, participant):
        pass


class DummyTurnDetector(TurnDetector):
    async def process_audio(self, audio_data, participant, conversation=None):
        pass


class DummyTTS(TTS):
    async def stream_audio(self, *_, **__):
        return b""

    async def stop_audio(self) -> None: ...


class DummyLLM(LLM, Warmable[bool]):
    def __init__(self):
        super(DummyLLM, self).__init__()
        self.warmed_up = False

    async def simple_response(self, *_, **__) -> LLMResponseEvent[Any]:
        return LLMResponseEvent(text="Simple response", original=None)

    async def on_warmup(self) -> bool:
        return True

    async def on_warmed_up(self, *_) -> None:
        self.warmed_up = True


class DummyEdge(EdgeTransport):
    def __init__(
        self,
        exc_on_join: Optional[Exception] = None,
        exc_on_publish_tracks: Optional[Exception] = None,
    ):
        super(DummyEdge, self).__init__()
        self.events = EventManager()
        self.exc_on_join = exc_on_join
        self.exc_on_publish_tracks = exc_on_publish_tracks
        self.authenticate_call_count = 0

    async def authenticate(self, user: User) -> None:
        self.authenticate_call_count += 1
        self._authenticated = True

    async def create_call(
        self, call_id: str, agent_user_id: Optional[str] = None, **kwargs
    ) -> Call:
        return DummyCall(call_id=call_id)

    def create_audio_track(self, *args, **kwargs) -> AudioStreamTrack:
        return AudioStreamTrack(
            audio_buffer_size_ms=300_000,
            sample_rate=48000,
            channels=2,
        )

    async def close(self):
        pass

    def open_demo(self, *args, **kwargs):
        pass

    async def join(self, *args, **kwargs):
        await asyncio.sleep(1)
        if self.exc_on_join:
            raise self.exc_on_join

    async def publish_tracks(self, audio_track, video_track):
        await asyncio.sleep(1)
        if self.exc_on_publish_tracks:
            raise self.exc_on_publish_tracks

    async def create_conversation(self, call: Any, user: User, instructions):
        pass

    def add_track_subscriber(self, track_id: str):
        pass

    async def send_custom_event(self, data: dict) -> None:
        self.last_custom_event = data


class DummyCall(Call):
    def __init__(self, call_id: str):
        self._id = call_id

    @property
    def id(self) -> str:
        return self._id


@pytest.fixture
def call():
    return DummyCall(call_id=str(uuid4()))


class SomeException(Exception):
    pass


class SlowSTT(BaseSTT):
    """STT whose ``start`` sleeps for a configurable delay before completing."""

    turn_detection: bool = False

    def __init__(self, delay: float) -> None:
        super().__init__()
        self._delay = delay

    async def start(self) -> None:
        await asyncio.sleep(self._delay)
        await super().start()

    async def process_audio(self, pcm_data, participant):
        pass


class SlowTurnDetector(TurnDetector):
    """Turn detector whose ``start`` sleeps and records completion."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self._delay = delay
        self.start_completed = False

    async def start(self) -> None:
        await asyncio.sleep(self._delay)
        self.start_completed = True
        await super().start()

    async def process_audio(self, data, participant, conversation=None):
        pass


class FailingSTT(BaseSTT):
    """STT that raises a configured exception in ``start`` or ``close``."""

    turn_detection: bool = False

    def __init__(
        self,
        *,
        exc_on_start: Optional[Exception] = None,
        exc_on_close: Optional[Exception] = None,
    ) -> None:
        super().__init__()
        self._exc_on_start = exc_on_start
        self._exc_on_close = exc_on_close

    async def start(self) -> None:
        if self._exc_on_start is not None:
            raise self._exc_on_start
        await super().start()

    async def close(self) -> None:
        if self._exc_on_close is not None:
            raise self._exc_on_close
        await super().close()

    async def process_audio(self, pcm_data, participant):
        pass


class WriteRecordingTrack:
    def __init__(self):
        self.writes: list[PcmData] = []

    async def write(self, data: PcmData) -> None:
        self.writes.append(data)


class DummyAudioPublisher(AudioPublisher):
    name = "dummy_audio"

    def __init__(self):
        super(DummyAudioPublisher, self).__init__()
        self.track = WriteRecordingTrack()

    def publish_audio_track(self) -> WriteRecordingTrack:
        return self.track

    async def close(self) -> None:
        pass


class RecordingEdge(DummyEdge):
    def __init__(self):
        super().__init__()
        self.recorded_audio_track = WriteRecordingTrack()

    def create_audio_track(self, *args, **kwargs) -> WriteRecordingTrack:
        return self.recorded_audio_track


class DummyAvatar(Avatar):
    def __init__(self) -> None:
        super().__init__()
        self._video_track = QueuedVideoTrack(width=640, height=480, fps=30)
        self._audio_output = AudioOutputStream()

    def video_output(self) -> aiortc.VideoStreamTrack:
        return self._video_track

    def audio_output(self) -> AudioOutputStream:
        return self._audio_output

    async def start(self) -> None: ...

    async def close(self) -> None: ...


class TestAgent:
    async def test_bare_final_marker_drains_output_track(self):
        # A bare end-of-turn marker (AudioOutputChunk with final=True but no data)
        # must drain the output track's resampler tail, so the utterance plays out
        # further than when no marker follows the audio.
        wave = (10000 * np.sin(2 * np.pi * 1000 * np.arange(4800) / 24000)).astype(
            np.int16
        )
        pcm = PcmData(samples=wave, sample_rate=24000, format="s16", channels=1)

        ends = []
        for send_marker in (False, True):
            agent = Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=DummyEdge(),
                agent_user=User(name="test"),
            )
            track = agent.audio_track
            producer = asyncio.create_task(agent._produce_audio_output())

            await agent._audio_output_stream.send(AudioOutputChunk(data=pcm))
            if send_marker:
                await agent._audio_output_stream.send(AudioOutputChunk(final=True))
            agent._audio_output_stream.close()
            await producer

            out = np.concatenate(
                [(await track.recv()).to_ndarray().reshape(-1) for _ in range(14)]
            )
            nonzero = np.nonzero(np.abs(out) > 1)[0]
            ends.append(int(nonzero[-1] + 1) if len(nonzero) else 0)

        assert ends[1] > ends[0]

    @pytest.mark.parametrize(
        "edge_params",
        [
            {"exc_on_join": SomeException("Test")},
            {"exc_on_publish_tracks": SomeException("Test")},
            {
                "exc_on_join": SomeException("Test"),
                "exc_on_publish_tracks": SomeException("Test"),
            },
        ],
    )
    async def test_join_suppress_exception_if_closing(self, call: Call, edge_params):
        """
        Test that errors during `Agent.join()` are suppressed if the agent is closing or already closed.
        """
        edge = DummyEdge(**edge_params)
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        # It must not fail because the agent is closing or already closed
        await asyncio.gather(agent.join(call).__aenter__(), agent.close())

    @pytest.mark.parametrize(
        "edge_params",
        [
            {"exc_on_join": SomeException("Test")},
            {"exc_on_publish_tracks": SomeException("Test")},
            {
                "exc_on_join": SomeException("Test"),
                "exc_on_publish_tracks": SomeException("Test"),
            },
        ],
    )
    async def test_join_propagates_exception(self, call: Call, edge_params):
        """
        Test that errors during `Agent.join()` are raised normally if the agent is not closing.
        """
        edge = DummyEdge(**edge_params)
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        with pytest.raises(SomeException):
            async with agent.join(call):
                ...

    async def test_start_components_runs_concurrently(self):
        """Components must be started in parallel, not sequentially."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=SlowSTT(delay=0.2),
            turn_detection=SlowTurnDetector(delay=0.2),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        t0 = time.perf_counter()
        await agent._start_components()
        elapsed = time.perf_counter() - t0
        # Two 200ms sleeps in parallel ≈ 0.2s; sequentially they would take ≈ 0.4s.
        assert elapsed < 0.35

    async def test_join_propagates_component_start_failure(self, call: Call):
        """A failing component ``start`` must surface as an error from ``join``."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=FailingSTT(exc_on_start=SomeException("boom")),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        with pytest.raises(SomeException):
            async with agent.join(call):
                ...

    async def test_stop_components_swallows_failures(self):
        """``_stop_components`` is best-effort: a failing component must not block others."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=FailingSTT(exc_on_close=SomeException("boom")),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        # Must not raise.
        await agent._stop_components()

    async def test_start_components_cancels_siblings_on_failure(self):
        """When one ``start`` raises, in-flight siblings must be cancelled."""
        slow = SlowTurnDetector(delay=1.0)
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=FailingSTT(exc_on_start=SomeException("boom")),
            turn_detection=slow,
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        with pytest.raises(SomeException):
            await agent._start_components()
        assert slow.start_completed is False

    async def test_send_custom_event(self):
        """Test that custom events are sent through the edge transport."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        test_data = {"type": "test_event", "value": 42}
        await agent.send_custom_event(test_data)

        assert edge.last_custom_event == test_data

    async def test_send_metrics_event(self):
        """Test that metrics are sent as custom events."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Update some metrics
        agent.metrics.llm_input_tokens__total.inc(100)
        agent.metrics.llm_output_tokens__total.inc(50)

        await agent.send_metrics_event()

        assert edge.last_custom_event["type"] == "agent_metrics"
        assert "metrics" in edge.last_custom_event
        assert edge.last_custom_event["metrics"]["llm_input_tokens__total"] == 100
        assert edge.last_custom_event["metrics"]["llm_output_tokens__total"] == 50

    async def test_send_metrics_event_with_fields_filter(self):
        """Test that only specified metric fields are included."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Update metrics
        agent.metrics.llm_input_tokens__total.inc(100)
        agent.metrics.tts_characters__total.inc(500)

        # Request only specific fields
        await agent.send_metrics_event(
            event_type="custom_metrics", fields=["llm_input_tokens__total"]
        )

        assert edge.last_custom_event["type"] == "custom_metrics"
        assert edge.last_custom_event["metrics"] == {"llm_input_tokens__total": 100}

    async def test_broadcast_metrics_enabled(self):
        """Test that metrics are automatically broadcast when enabled."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
            broadcast_metrics=True,
            broadcast_metrics_interval=0.1,  # Short interval for testing
        )

        # Update some metrics
        agent.metrics.llm_input_tokens__total.inc(42)

        # Start the broadcast task manually (normally happens during join)
        agent._metrics_broadcast_task = asyncio.create_task(
            agent._metrics_broadcast_loop()
        )

        # Wait for at least one broadcast
        await asyncio.sleep(0.15)

        # Cancel the task
        agent._metrics_broadcast_task.cancel()
        try:
            await agent._metrics_broadcast_task
        except asyncio.CancelledError:
            pass

        # Verify metrics were broadcast
        assert edge.last_custom_event["type"] == "agent_metrics"
        assert edge.last_custom_event["metrics"]["llm_input_tokens__total"] == 42

    async def test_audio_track_from_publisher(self):
        publisher = DummyAudioPublisher()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            processors=[publisher],
        )
        assert agent.audio_track is publisher.track

    async def test_audio_track_from_edge_without_publisher(self):
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        assert agent.audio_track is not None
        assert not agent.audio_publishers

    async def test_audio_publishers_property(self):
        publisher = DummyAudioPublisher()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            processors=[publisher],
        )
        assert agent.audio_publishers == [publisher]

    async def test_authenticate_calls_edge(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        await agent.authenticate()
        assert edge.authenticate_call_count == 1

    async def test_authenticate_is_idempotent(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        await agent.authenticate()
        await agent.authenticate()
        await agent.authenticate()
        assert edge.authenticate_call_count == 1

    async def test_create_call_authenticates_automatically(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        call = await agent.create_call("default", "call-1")
        assert call.id == "call-1"
        assert edge.authenticate_call_count == 1

    async def test_create_call_does_not_double_authenticate(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        await agent.authenticate()
        await agent.create_call("default", "call-1")
        assert edge.authenticate_call_count == 1

    async def test_join_authenticates_automatically(self, call: Call):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        async with agent.join(call):
            assert edge.authenticate_call_count == 1

    async def test_join_does_not_double_authenticate(self, call: Call):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        await agent.authenticate()
        async with agent.join(call):
            assert edge.authenticate_call_count == 1

    async def test_avatar_wiring(self):
        """Avatar metrics forward to agent metrics after merge, and the
        avatar's video track becomes the agent's outbound video track."""
        avatar = DummyAvatar()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            avatar=avatar,
        )
        avatar.metrics.on_llm_response(input_tokens=7)
        assert agent.metrics.llm_input_tokens__total.value() == 7
        assert agent._video_track is avatar.video_output()

    async def test_publish_video_true_with_avatar_false_without(self):
        agent_with = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            avatar=DummyAvatar(),
        )
        agent_without = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        assert agent_with.publish_video is True
        assert agent_without.publish_video is False
