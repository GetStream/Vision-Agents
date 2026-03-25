import asyncio
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pytest
from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core import Agent, User
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge import Call, EdgeTransport
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeAudioOutputEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors.base_processor import AudioPublisher
from vision_agents.core.stt import STT as BaseSTT
from vision_agents.core.tts import TTS
from vision_agents.core.tts.events import TTSAudioEvent
from vision_agents.core.turn_detection import TurnDetector
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


class WriteRecordingTrack:
    def __init__(self):
        self.writes: list[PcmData] = []

    async def write(self, data: PcmData) -> None:
        self.writes.append(data)


class DummyAudioPublisher(AudioPublisher):
    name = "dummy_audio"

    def __init__(self):
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


class TestAgent:
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

    async def test_tts_audio_not_forwarded_with_publisher(self):
        publisher = DummyAudioPublisher()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            processors=[publisher],
        )
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
        )
        agent.events.send(TTSAudioEvent(data=pcm))
        await agent.events.wait()
        assert publisher.track.writes == []

    async def test_tts_audio_forwarded_without_publisher(self):
        edge = RecordingEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
        )
        agent.events.send(TTSAudioEvent(data=pcm))
        await agent.events.wait()
        assert len(edge.recorded_audio_track.writes) == 1
        assert edge.recorded_audio_track.writes[0] is pcm

    async def test_realtime_audio_not_forwarded_with_publisher(self):
        publisher = DummyAudioPublisher()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
            processors=[publisher],
        )
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
        )
        agent.events.send(RealtimeAudioOutputEvent(data=pcm))
        await agent.events.wait()
        assert publisher.track.writes == []

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

    async def test_stale_tts_epoch_dropped(self):
        edge = RecordingEdge()
        tts = DummyTTS()
        agent = Agent(
            llm=DummyLLM(),
            tts=tts,
            edge=edge,
            agent_user=User(name="test"),
        )
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
        )
        await tts.interrupt()
        agent.events.send(TTSAudioEvent(data=pcm, epoch=0))
        await agent.events.wait()
        assert edge.recorded_audio_track.writes == []

    async def test_current_tts_epoch_forwarded(self):
        edge = RecordingEdge()
        tts = DummyTTS()
        agent = Agent(
            llm=DummyLLM(),
            tts=tts,
            edge=edge,
            agent_user=User(name="test"),
        )
        pcm = PcmData(
            samples=np.zeros(160, dtype=np.int16),
            sample_rate=16000,
            format=AudioFormat.S16,
        )
        await tts.interrupt()
        agent.events.send(TTSAudioEvent(data=pcm, epoch=tts.epoch))
        await agent.events.wait()
        assert len(edge.recorded_audio_track.writes) == 1

    async def test_stt_turn_detection_ignores_external_turn_detector(self):
        """When STT provides turn detection, the external TurnDetector is ignored."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=DummySTTWithTurnDetection(),
            turn_detection=DummyTurnDetector(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        assert agent.turn_detection is None
        assert agent.turn_detection_enabled is True

    async def test_stt_without_turn_detection_keeps_external_turn_detector(self):
        """When STT does not provide turn detection, the external TurnDetector is kept."""
        detector = DummyTurnDetector()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=DummySTT(),
            turn_detection=detector,
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        assert agent.turn_detection is detector
        assert agent.turn_detection_enabled is True

    async def test_stt_turn_detection_without_external_detector(self):
        """STT with turn detection works without an external TurnDetector."""
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            stt=DummySTTWithTurnDetection(),
            edge=DummyEdge(),
            agent_user=User(name="test"),
        )
        assert agent.turn_detection is None
        assert agent.turn_detection_enabled is True


@pytest.fixture
def participant() -> Participant:
    return Participant(id="participant-user-1", user_id="user-1", original=None)


@pytest.fixture
async def transcript_agent() -> Agent:
    agent = Agent(
        llm=DummyLLM(),
        tts=DummyTTS(),
        edge=DummyEdge(),
        agent_user=User(id="agent-1", name="bot"),
    )
    agent.conversation = InMemoryConversation(instructions="", messages=[])
    return agent


async def _send(agent: Agent, event) -> None:
    agent.events.send(event)
    await agent.events.wait(1.0)


class TestAgentTranscriptHandling:
    """Tests for accumulating realtime transcript events into conversation messages."""

    async def test_final_user_transcript_creates_single_message(
        self, transcript_agent, participant
    ):
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello world", mode="final", participant=participant
            ),
        )

        assert len(transcript_agent.conversation.messages) == 1
        assert transcript_agent.conversation.messages[0].content == "Hello world"

    async def test_replacement_user_transcripts_reuse_message_id(
        self, transcript_agent, participant
    ):
        for text in ["Hello", "Hello world", "Hello world how"]:
            await _send(
                transcript_agent,
                RealtimeUserSpeechTranscriptionEvent(
                    text=text, mode="replacement", participant=participant
                ),
            )

        assert len(transcript_agent.conversation.messages) == 1
        assert transcript_agent.conversation.messages[0].content == "Hello world how"

    async def test_final_after_replacements_reuses_same_message(
        self, transcript_agent, participant
    ):
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="Hi", mode="replacement", participant=participant
            ),
        )
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="Hi there", mode="final", participant=participant
            ),
        )

        assert len(transcript_agent.conversation.messages) == 1
        assert transcript_agent.conversation.messages[0].content == "Hi there"

    async def test_agent_delta_transcripts_reuse_message_id(self, transcript_agent):
        for text in ["I'm ", "doing ", "well"]:
            await _send(
                transcript_agent,
                RealtimeAgentSpeechTranscriptionEvent(text=text, mode="delta"),
            )

        assert len(transcript_agent.conversation.messages) == 1
        assert transcript_agent.conversation.messages[0].content == "I'm doing well"

    async def test_agent_final_transcript_reuses_delta_message(self, transcript_agent):
        await _send(
            transcript_agent,
            RealtimeAgentSpeechTranscriptionEvent(text="Thinking", mode="delta"),
        )
        await _send(
            transcript_agent,
            RealtimeAgentSpeechTranscriptionEvent(text="", mode="final"),
        )

        assert len(transcript_agent.conversation.messages) == 1
        assert transcript_agent.conversation.messages[0].content == "Thinking"

    async def test_user_transcript_finalizes_pending_agent(
        self, transcript_agent, participant
    ):
        """When user starts speaking, any pending agent transcript is finalized."""
        await _send(
            transcript_agent,
            RealtimeAgentSpeechTranscriptionEvent(text="Agent response", mode="delta"),
        )
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="User reply", mode="delta", participant=participant
            ),
        )

        assert len(transcript_agent.conversation.messages) == 2
        assert transcript_agent.conversation.messages[0].role == "assistant"
        assert transcript_agent.conversation.messages[1].role == "user"
        assert transcript_agent.transcripts.flush_agent_transcript() is None

    async def test_agent_transcript_finalizes_pending_user(
        self, transcript_agent, participant
    ):
        """When agent starts speaking, any pending user transcript is finalized."""
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="User speaking", mode="delta", participant=participant
            ),
        )
        await _send(
            transcript_agent,
            RealtimeAgentSpeechTranscriptionEvent(text="Agent reply", mode="delta"),
        )

        assert len(transcript_agent.conversation.messages) == 2
        assert transcript_agent.conversation.messages[0].role == "user"
        assert transcript_agent.conversation.messages[1].role == "assistant"
        assert not transcript_agent.transcripts.flush_users_transcripts()

    async def test_multiple_turns_create_separate_messages(
        self, transcript_agent, participant
    ):
        """Full conversation flow: user -> agent -> user."""
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="Hi", mode="final", participant=participant
            ),
        )
        await _send(
            transcript_agent,
            RealtimeAgentSpeechTranscriptionEvent(text="Hello!", mode="final"),
        )
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="How are you?", mode="final", participant=participant
            ),
        )

        assert len(transcript_agent.conversation.messages) == 3
        assert transcript_agent.conversation.messages[0].content == "Hi"
        assert transcript_agent.conversation.messages[1].content == "Hello!"
        assert transcript_agent.conversation.messages[2].content == "How are you?"

    async def test_gemini_style_delta_transcripts(self, transcript_agent, participant):
        """Gemini sends incremental delta chunks."""
        for text in ["I ", "am ", "walking ", "to ", "the store"]:
            await _send(
                transcript_agent,
                RealtimeUserSpeechTranscriptionEvent(
                    text=text, mode="delta", participant=participant
                ),
            )

        assert len(transcript_agent.conversation.messages) == 1
        assert (
            transcript_agent.conversation.messages[0].content
            == "I am walking to the store"
        )

    async def test_openai_style_final_transcripts(self, transcript_agent, participant):
        """OpenAI sends a single final transcript (no deltas)."""
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="OK, everybody. Do.",
                mode="final",
                participant=participant,
            ),
        )

        assert len(transcript_agent.conversation.messages) == 1
        assert transcript_agent.conversation.messages[0].content == "OK, everybody. Do."

    async def test_no_participant_skips_sync(self, transcript_agent):
        """Events without a participant should not create messages."""
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="orphan transcript", mode="final"
            ),
        )

        assert len(transcript_agent.conversation.messages) == 0

    async def test_empty_text_skips_sync(self, transcript_agent, participant):
        """Events with empty text should not create messages."""
        await _send(
            transcript_agent,
            RealtimeUserSpeechTranscriptionEvent(
                text="", mode="final", participant=participant
            ),
        )

        assert len(transcript_agent.conversation.messages) == 0
