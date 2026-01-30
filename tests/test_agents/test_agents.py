import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from getstream.video.rtc import Call
from vision_agents.core import Agent, User
from vision_agents.core.edge import EdgeTransport
from vision_agents.core.edge.types import OutputAudioTrack, Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeUserSpeechTranscriptionEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.tts import TTS
from vision_agents.core.utils import audio_track
from vision_agents.core.warmup import Warmable


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

    async def create_user(self, user: User):
        return

    def create_audio_track(self, *args, **kwargs) -> OutputAudioTrack:
        return audio_track.AudioStreamTrack(
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


@pytest.fixture
def call():
    return Call(call_id=str(uuid4()), call_type="default", client=AsyncMock())


class SomeException(Exception):
    pass


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


class DummyConversation:
    """Minimal conversation mock for testing."""

    def __init__(self):
        self.messages = {}

    async def upsert_message(self, **kwargs):
        message_id = kwargs.get("message_id")
        if message_id:
            self.messages[message_id] = kwargs


class TestRealtimeTranscriptAccumulation:
    """Tests for realtime transcript accumulation behavior."""

    async def test_user_transcripts_accumulate_into_single_message(self):
        """Test that multiple user transcript events accumulate into a single message."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Mock conversation to allow event processing
        agent.conversation = DummyConversation()

        participant = Participant(original=None, user_id="user-123")

        # Simulate multiple incremental transcript events (like Gemini sends)
        events = [
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello", participant=participant, plugin_name="test"
            ),
            RealtimeUserSpeechTranscriptionEvent(
                text="how", participant=participant, plugin_name="test"
            ),
            RealtimeUserSpeechTranscriptionEvent(
                text="are", participant=participant, plugin_name="test"
            ),
            RealtimeUserSpeechTranscriptionEvent(
                text="you", participant=participant, plugin_name="test"
            ),
        ]

        # Send all events
        for event in events:
            agent.events.send(event)

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify that there's only ONE message ID tracked for this user
        assert len(agent._realtime_user_message_ids) == 1
        assert "user-123" in agent._realtime_user_message_ids

        # Verify the accumulated text
        assert agent._realtime_user_accumulated_text["user-123"] == "Hello how are you"

        # Verify that only ONE message was created in the conversation
        assert len(agent.conversation.messages) == 1

        # Verify the final message content
        message_id = agent._realtime_user_message_ids["user-123"]
        assert agent.conversation.messages[message_id]["content"] == "Hello how are you"

    async def test_user_transcripts_reset_after_agent_response(self):
        """Test that user transcript tracking resets when agent responds."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Mock conversation
        agent.conversation = DummyConversation()

        participant = Participant(original=None, user_id="user-123")

        # Send user transcript
        user_event = RealtimeUserSpeechTranscriptionEvent(
            text="Hello", participant=participant, plugin_name="test"
        )
        agent.events.send(user_event)
        await asyncio.sleep(0.05)

        # Verify tracking is active
        assert "user-123" in agent._realtime_user_message_ids
        original_message_id = agent._realtime_user_message_ids["user-123"]

        # Agent responds
        agent_event = RealtimeAgentSpeechTranscriptionEvent(
            text="Hi there!", plugin_name="test"
        )
        agent.events.send(agent_event)
        await asyncio.sleep(0.05)

        # Verify tracking was reset
        assert len(agent._realtime_user_message_ids) == 0
        assert len(agent._realtime_user_accumulated_text) == 0

        # New user message should get a new message ID
        user_event2 = RealtimeUserSpeechTranscriptionEvent(
            text="Thanks", participant=participant, plugin_name="test"
        )
        agent.events.send(user_event2)
        await asyncio.sleep(0.05)

        assert "user-123" in agent._realtime_user_message_ids
        new_message_id = agent._realtime_user_message_ids["user-123"]

        # Message IDs should be different (new turn)
        assert new_message_id != original_message_id

    async def test_multiple_users_tracked_separately(self):
        """Test that transcripts from different users are tracked separately."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Mock conversation
        agent.conversation = DummyConversation()

        participant1 = Participant(original=None, user_id="user-1")
        participant2 = Participant(original=None, user_id="user-2")

        # User 1 speaks
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello", participant=participant1, plugin_name="test"
            )
        )
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="world", participant=participant1, plugin_name="test"
            )
        )

        # User 2 speaks
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="Hi", participant=participant2, plugin_name="test"
            )
        )
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="there", participant=participant2, plugin_name="test"
            )
        )

        await asyncio.sleep(0.1)

        # Verify each user has their own tracking
        assert len(agent._realtime_user_message_ids) == 2
        assert agent._realtime_user_accumulated_text["user-1"] == "Hello world"
        assert agent._realtime_user_accumulated_text["user-2"] == "Hi there"

        # Verify different message IDs for each user
        assert (
            agent._realtime_user_message_ids["user-1"]
            != agent._realtime_user_message_ids["user-2"]
        )

        # Verify two separate messages in conversation
        assert len(agent.conversation.messages) == 2

    async def test_already_accumulated_transcripts_not_duplicated(self):
        """Test that already-accumulated transcripts (like from OpenAI) don't get duplicated."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Mock conversation
        agent.conversation = DummyConversation()

        participant = Participant(original=None, user_id="user-123")

        # Simulate accumulated transcripts (each event contains full text so far)
        # This is how some APIs might send transcripts
        events = [
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello", participant=participant, plugin_name="test"
            ),
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello how", participant=participant, plugin_name="test"
            ),
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello how are", participant=participant, plugin_name="test"
            ),
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello how are you", participant=participant, plugin_name="test"
            ),
        ]

        # Send all events
        for event in events:
            agent.events.send(event)

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify the final text is NOT duplicated (not "Hello Hello how Hello how are...")
        assert agent._realtime_user_accumulated_text["user-123"] == "Hello how are you"

        # Verify only one message
        assert len(agent.conversation.messages) == 1

    async def test_mixed_delta_and_complete_transcripts(self):
        """Test handling a mix of delta and complete transcripts."""
        edge = DummyEdge()
        agent = Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=edge,
            agent_user=User(name="test"),
        )

        # Mock conversation
        agent.conversation = DummyConversation()

        participant = Participant(original=None, user_id="user-123")

        # First turn: user speaks (deltas)
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="Hello", participant=participant, plugin_name="test"
            )
        )
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="there", participant=participant, plugin_name="test"
            )
        )
        await asyncio.sleep(0.05)

        # Verify first turn accumulated
        assert agent._realtime_user_accumulated_text["user-123"] == "Hello there"
        first_message_id = agent._realtime_user_message_ids["user-123"]

        # Agent responds - resets tracking
        agent.events.send(
            RealtimeAgentSpeechTranscriptionEvent(text="Hi!", plugin_name="test")
        )
        await asyncio.sleep(0.05)

        # Second turn: user speaks (complete transcript from OpenAI-style API)
        agent.events.send(
            RealtimeUserSpeechTranscriptionEvent(
                text="How are you doing today?",
                participant=participant,
                plugin_name="test",
            )
        )
        await asyncio.sleep(0.05)

        # Verify second turn is a new message
        assert (
            agent._realtime_user_accumulated_text["user-123"]
            == "How are you doing today?"
        )
        second_message_id = agent._realtime_user_message_ids["user-123"]

        # Different message IDs for different turns
        assert first_message_id != second_message_id
