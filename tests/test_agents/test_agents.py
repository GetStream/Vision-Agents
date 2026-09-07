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
from vision_agents.core.agents.agents import DEFAULT_INSTRUCTIONS
from vision_agents.core.agents.events import UserTranscriptEvent
from vision_agents.core.agents.inference import AudioOutputChunk, AudioOutputStream
from vision_agents.core.avatars import Avatar
from vision_agents.core.edge import Call, EdgeTransport
from vision_agents.core.events import EventManager
from vision_agents.core.harness import DefaultHarness
from vision_agents.core.llm.llm import LLM, LLMResponseEvent, OmniLLM
from vision_agents.core.llm.remote import RemoteCall, RemoteEvent
from vision_agents.core.processors.base_processor import AudioPublisher
from vision_agents.core.stt import STT as BaseSTT
from vision_agents.core.telephony import InboundCall, OutboundCall, PlacedCall
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
    model = "tts"

    async def stream_audio(self, *_, **__):
        return b""

    async def stop_audio(self) -> None: ...


class DummyLLM(LLM, Warmable[bool]):
    model = "llm"

    def __init__(self):
        super(DummyLLM, self).__init__()
        self.warmed_up = False

    async def simple_response(self, *_, **__) -> LLMResponseEvent[Any]:
        return LLMResponseEvent(text="Simple response", original=None)

    async def on_warmup(self) -> bool:
        return True

    async def on_warmed_up(self, *_) -> None:
        self.warmed_up = True


class DummyPhone:
    """Somewhere to place a call that records what it was asked to place."""

    def __init__(self):
        self.placed: Optional[OutboundCall] = None

    async def place(self, call: OutboundCall) -> PlacedCall:
        self.placed = call
        return PlacedCall(
            vendor_call_id="CA123",
            status="queued",
            vendor="twilio",
            call_id=call.call_id,
            call_type=call.call_type,
        )


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
        self.created_calls: list[str] = []

    async def authenticate(self, user: User) -> None:
        self.authenticate_call_count += 1
        self._authenticated = True

    async def create_call(
        self, call_id: str, agent_user_id: Optional[str] = None, **kwargs
    ) -> Call:
        self.created_calls.append(call_id)
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


class DummyRemotePipeline(OmniLLM):
    """An LLM whose pipeline runs somewhere else, as far as the agent can tell."""

    model = "remote"

    def __init__(self):
        super().__init__()
        self.joined: Optional[RemoteCall] = None
        self.left = False
        self.said: list[str] = []
        self.session_id: Optional[str] = None
        self._reported: asyncio.Queue[Optional[RemoteEvent]] = asyncio.Queue()

    @property
    def router_session_id(self) -> Optional[str]:
        return self.session_id

    async def join_remote(self, call: RemoteCall) -> None:
        self.joined = call

    async def remote_events(self):
        while True:
            event = await self._reported.get()
            if event is None:
                return
            yield event

    async def say_remote(self, text: str, interrupt: bool = False) -> None:
        self.said.append(text)

    async def respond_remote(self, text: str, interrupt: bool = True) -> None:
        self.said.append(text)

    async def leave_remote(self) -> None:
        self.left = True
        await self._reported.put(None)

    async def report(self, event: RemoteEvent) -> None:
        """Say that the remote pipeline did something."""
        await self._reported.put(event)

    async def simple_response(self, *_, **__):
        return
        yield

    async def simple_audio_response(self, pcm, participant):
        pass

    async def watch_video_track(self, track, shared_forwarder=None) -> None:
        pass

    async def stop_watching_video_track(self) -> None:
        pass


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

    async def test_authenticate_attaches_agent_metadata(self):
        """authenticate() records provider and model metadata on the agent user."""
        llm = DummyLLM()
        llm.model = "dummy-llm-model"
        stt = DummySTT()
        stt.model = "dummy-stt-model"

        tts = DummyTTS()
        tts.model = None

        agent = Agent(
            llm=llm,
            stt=stt,
            tts=tts,
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        await agent.authenticate()

        custom = agent.agent_user.custom
        assert custom["is_agent"] is True
        assert custom["llm"] == {"provider": "DummyLLM", "model": "dummy-llm-model"}
        assert custom["stt"] == {"provider": "DummySTT", "model": "dummy-stt-model"}
        # No model set on the TTS -> provider only.
        assert custom["tts"] == {"provider": "DummyTTS"}

    async def test_authenticate_preserves_user_supplied_custom(self):
        """Non-managed user custom keys survive; framework owns component metadata."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test", custom={"is_agent": False, "team": "red"}),
        )

        await agent.authenticate()

        custom = agent.agent_user.custom
        assert custom["is_agent"] is True
        assert custom["team"] == "red"
        assert custom["llm"] == {"provider": "DummyLLM", "model": "llm"}

    async def test_authenticate_clears_stale_managed_custom(self):
        """Stale component keys in agent_user.custom cannot override current metadata."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(
                name="test",
                custom={
                    "stt": {"provider": "deepgram", "model": "flux-general-en"},
                    "team": "red",
                },
            ),
        )

        await agent.authenticate()

        custom = agent.agent_user.custom
        assert custom["stt"] is None
        assert custom["team"] == "red"
        assert custom["llm"] == {"provider": "DummyLLM", "model": "llm"}
        assert custom["tts"] == {"provider": "DummyTTS", "model": "tts"}
        assert custom["is_agent"] is True

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

    async def test_an_outbound_call_rings_the_person_and_joins_where_they_land(self):
        # The call exists and the agent is in it before the phone rings, so nobody
        # answers to silence.
        phone = DummyPhone()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            phone=phone,
        )

        async with agent.outbound_call(
            from_="+17195551234",
            to="+13035559876",
            call_id="support-line",
            ring_timeout=20.0,
            initial_digits="ww1234#",
            wait_for_end=False,
        ) as placed:
            assert agent.call is not None
            assert agent.call.id == "support-line"
            assert placed.vendor_call_id == "CA123"

        assert phone.placed is not None
        assert phone.placed.from_ == "+17195551234"
        assert phone.placed.to == "+13035559876"
        # The call the vendor is told to bridge into is the one just created, or the
        # person answers into an empty room.
        assert phone.placed.call_id == "support-line"
        assert phone.placed.ring_timeout == 20.0
        assert phone.placed.initial_digits == "ww1234#"

    async def test_an_outbound_call_without_a_phone_says_what_to_pass(self):
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        with pytest.raises(ValueError, match="phone="):
            async with agent.outbound_call(from_="+17195551234", to="+13035559876"):
                pass

    async def test_answering_joins_the_call_the_caller_is_already_in(self):
        # The caller reached the call over SIP before anything here knew about it, so
        # creating a second one would leave the agent talking to an empty room.
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        arriving = InboundCall(
            call_id="phone-+15125551234",
            call_type="default",
            called_number="+15125551234",
            caller_number="+15550001111",
        )

        async with agent.answer(arriving, wait_for_end=False):
            assert agent.call is not None
            assert agent.call.id == "phone-+15125551234"

        assert edge.created_calls == ["phone-+15125551234"]

    async def test_answering_uses_the_call_type_the_call_arrived_on(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        async with agent.answer(
            InboundCall(call_id="the-support-line", call_type="support"),
            wait_for_end=False,
        ):
            assert agent._call_type == "support"

    async def test_answering_a_call_that_names_no_call_is_refused(self):
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        with pytest.raises(ValueError, match="names? the call"):
            async with agent.answer(InboundCall(call_id="")):
                pass

    async def test_joining_an_inbound_call_attaches_to_the_call_the_caller_is_in(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        arriving = InboundCall(
            call_id="phone-+15125551234",
            call_type="default",
            called_number="+15125551234",
        )

        async with agent.join(arriving, wait_for_end=False):
            assert agent.call is not None
            assert agent.call.id == "phone-+15125551234"
            await arriving.wait_for_phone_participant()

        assert edge.created_calls == ["phone-+15125551234"]

    async def test_waiting_for_the_caller_before_joining_says_so(self):
        arriving = InboundCall(call_id="phone-+15125551234")

        with pytest.raises(RuntimeError, match="join the call"):
            await arriving.wait_for_phone_participant()

    async def test_an_agent_without_an_llm_needs_a_config(self):
        with pytest.raises(ValueError, match="config="):
            Agent()

    async def test_an_agent_nobody_instructed_gets_the_default(self):
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        assert agent.instructions.full_reference == DEFAULT_INSTRUCTIONS

    async def test_an_agent_built_from_a_config_is_instructed_by_it(self):
        # The config carries its own instructions, and the backend lets anything the
        # agent says override them, so a default here would replace what it says.
        agent = Agent(
            config="support",
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        assert agent.instructions.full_reference == ""

    async def test_instructions_passed_alongside_a_config_still_win(self):
        agent = Agent(
            config="support",
            instructions="Only speak Dutch.",
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        assert agent.instructions.full_reference == "Only speak Dutch."

    async def test_join_authenticates_automatically(self, call: Call):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        async with agent.join(call, wait_for_end=False):
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
        async with agent.join(call, wait_for_end=False):
            assert edge.authenticate_call_count == 1

    async def test_joining_a_call_type_and_an_id_creates_the_call(self):
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        async with agent.join("default", "call-9", wait_for_end=False):
            pass

        assert edge.created_calls == ["call-9"]
        assert agent.call is not None
        assert agent.call.id == "call-9"

    async def test_joining_a_call_type_without_an_id_is_refused(self):
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        with pytest.raises(ValueError, match="needs the call's id"):
            async with agent.join("default", wait_for_end=False):
                pass

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

    async def test_agent_components_metadata_positive(self):
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            avatar=DummyAvatar(),
        )
        assert agent.components_metadata == {
            "llm": {"provider": "DummyLLM", "model": "llm"},
            "tts": {"provider": "DummyTTS", "model": "tts"},
            "avatar": {"provider": "DummyAvatar"},
        }

    async def test_agent_components_metadata_model_is_not_str(self):
        class LLMWithModelObject(DummyLLM):
            model = object()

        agent = Agent(
            llm=LLMWithModelObject(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            avatar=DummyAvatar(),
        )
        assert agent.components_metadata == {
            "llm": {
                "provider": "LLMWithModelObject",
            },
            "tts": {"provider": "DummyTTS", "model": "tts"},
            "avatar": {"provider": "DummyAvatar"},
        }

    async def test_join_hands_the_call_to_a_remote_pipeline(self, call: Call):
        llm = DummyRemotePipeline()
        agent = Agent(
            llm=llm,
            edge=DummyEdge(),
            agent_user=User(name="test"),
            instructions="be brief",
            harness=DefaultHarness(subagents={"default": "llm-smart"}),
            cost_tracking={"project": "moderation"},
            memory_filter={"user_id": "222"},
        )

        async with agent.join(call, wait_for_end=False):
            pass

        assert llm.joined is not None
        assert llm.joined.call_id == call.id
        assert llm.joined.instructions == "be brief"
        assert llm.joined.harness is not None
        assert llm.joined.harness.subagent == "llm-smart"
        assert llm.joined.cost_tracking == {"project": "moderation"}
        assert llm.joined.memory_filter == {"user_id": "222"}
        assert llm.left

    async def test_a_remote_pipeline_reports_speech_as_the_agents_own_events(
        self, call: Call
    ):
        llm = DummyRemotePipeline()
        agent = Agent(llm=llm, edge=DummyEdge(), agent_user=User(name="test"))

        heard: list[UserTranscriptEvent] = []
        arrived = asyncio.Event()

        @agent.subscribe
        async def on_transcript(event: UserTranscriptEvent):
            heard.append(event)
            arrived.set()

        async with agent.join(call, wait_for_end=False):
            await llm.report(
                RemoteEvent(
                    type="user_speech",
                    text="hello there",
                    user_id="u1",
                    participant_id="p1",
                )
            )
            await asyncio.wait_for(arrived.wait(), timeout=5)

        assert [event.text for event in heard] == ["hello there"]
        assert heard[0].participant is not None
        assert heard[0].participant.user_id == "u1"

    async def test_saying_something_on_a_remote_call_goes_to_the_pipeline(
        self, call: Call
    ):
        llm = DummyRemotePipeline()
        agent = Agent(llm=llm, edge=DummyEdge(), agent_user=User(name="test"))

        async with agent.join(call, wait_for_end=False):
            await agent.say("one moment")
            await agent.simple_response("greet them")

        assert llm.said == ["one moment", "greet them"]

    async def test_leaving_the_block_waits_for_the_call_to_end(self, call: Call):
        # Leaving the block is not hanging up: an agent that has said its greeting stays
        # until the call is over, so nothing has to be waited on by hand.
        llm = DummyRemotePipeline()
        agent = Agent(llm=llm, edge=DummyEdge(), agent_user=User(name="test"))
        ended = False

        async def end_the_call():
            nonlocal ended
            await asyncio.sleep(0.05)
            ended = True
            await llm.report(RemoteEvent(type="ended"))

        ending = asyncio.create_task(end_the_call())
        async with agent.join(call, participant_wait_timeout=0):
            assert not ended

        assert ended
        await ending

    async def test_a_call_can_be_left_without_waiting_for_it_to_end(self, call: Call):
        llm = DummyRemotePipeline()
        agent = Agent(llm=llm, edge=DummyEdge(), agent_user=User(name="test"))

        async with agent.join(call, wait_for_end=False):
            pass

        assert llm.left


class TestOpenUI:
    """How `run` decides whether there is a dashboard page to open."""

    async def test_a_remote_pipeline_that_has_joined_names_the_page(self):
        from vision_agents.core.runner.runner import _router_session_id

        llm = DummyRemotePipeline()
        llm.session_id = "call-page"
        agent = Agent(llm=llm, edge=DummyEdge(), agent_user=User(name="test"))

        assert await _router_session_id(agent) == "call-page"

    async def test_a_join_that_already_failed_does_not_hold_the_ui(self):
        from vision_agents.core.runner.runner import _router_session_id

        llm = DummyRemotePipeline()
        agent = Agent(llm=llm, edge=DummyEdge(), agent_user=User(name="test"))

        async def fail() -> None:
            raise RuntimeError("the call was not created")

        task = asyncio.create_task(fail())
        # The task finishes on its own; waiting on it would raise and is not what
        # opening the UI does.
        while not task.done():
            await asyncio.sleep(0)
        task.exception()
        started = time.monotonic()

        assert await _router_session_id(agent, join_task=task) is None
        assert time.monotonic() - started < 1

    async def test_a_local_pipeline_has_no_dashboard_page(self):
        from vision_agents.core.runner.runner import _router_session_id

        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )

        assert await _router_session_id(agent) is None
