from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2
from vision_agents.core.events.manager import EventManager
from vision_agents.plugins.getstream.sfu_events import (
    AudioLevelEvent,
    ParticipantJoinedEvent,
    TrackPublishedEvent,
    TrackUnpublishedEvent,
)


class TestSFUEvents:
    """Tests for SFU events in the GetStream plugin."""

    async def test_protobuf_events_with_base_event(self):
        """Test that event manager handles protobuf events that inherit from BaseEvent."""

        manager = EventManager()

        # Register generated protobuf event classes
        manager.register(AudioLevelEvent)
        manager.register(ParticipantJoinedEvent)

        assert AudioLevelEvent.type in manager._events
        assert ParticipantJoinedEvent.type in manager._events

        # Test 1: Send wrapped protobuf event with BaseEvent fields
        proto_audio = events_pb2.AudioLevel(
            user_id="user123", level=0.85, is_speaking=True
        )
        wrapped_event = AudioLevelEvent.from_proto(proto_audio, session_id="session123")

        received_audio_events = []

        @manager.subscribe
        async def handle_audio(event: AudioLevelEvent):
            received_audio_events.append(event)

        manager.send(wrapped_event)
        await manager.wait()

        assert len(received_audio_events) == 1
        assert received_audio_events[0].user_id == "user123"
        assert received_audio_events[0].session_id == "session123"
        assert received_audio_events[0].is_speaking is True
        assert received_audio_events[0].level is not None
        assert abs(received_audio_events[0].level - 0.85) < 0.01
        assert hasattr(received_audio_events[0], "event_id")
        assert hasattr(received_audio_events[0], "timestamp")

        # Test 2: Send raw protobuf message (auto-wrapped)
        proto_raw = events_pb2.AudioLevel(
            user_id="user456", level=0.95, is_speaking=False
        )

        received_audio_events.clear()
        manager.send(proto_raw)
        await manager.wait()

        assert len(received_audio_events) == 1
        assert received_audio_events[0].user_id == "user456"
        assert received_audio_events[0].level is not None
        assert abs(received_audio_events[0].level - 0.95) < 0.01
        assert received_audio_events[0].is_speaking is False
        assert hasattr(received_audio_events[0], "event_id")

        # Test 3: Create event without protobuf payload (all fields optional)
        empty_event = AudioLevelEvent()
        assert empty_event.payload is None
        assert empty_event.user_id is None
        assert empty_event.event_id is not None

        # Test 4: Multiple protobuf event types
        received_participant_events = []

        @manager.subscribe
        async def handle_participant(event: ParticipantJoinedEvent):
            received_participant_events.append(event)

        participant = models_pb2.Participant(user_id="user789", session_id="sess456")
        proto_participant = events_pb2.ParticipantJoined(
            call_cid="call123", participant=participant
        )

        manager.send(proto_participant)
        await manager.wait()

        assert len(received_participant_events) == 1
        assert received_participant_events[0].call_cid == "call123"
        assert received_participant_events[0].participant is not None
        assert hasattr(received_participant_events[0], "event_id")

    async def test_track_published_event_with_participant_property(self):
        """Test that TrackPublishedEvent correctly handles participant property override."""

        manager = EventManager()

        # Register events that override participant field with property
        manager.register(TrackPublishedEvent)
        manager.register(TrackUnpublishedEvent)

        # Test TrackPublishedEvent
        participant = models_pb2.Participant(user_id="user123", session_id="session456")
        proto_published = events_pb2.TrackPublished(
            user_id="user123", participant=participant
        )

        # This should NOT raise "AttributeError: property 'participant' of 'TrackPublishedEvent' object has no setter"
        TrackPublishedEvent.from_proto(proto_published)

        received_events = []

        @manager.subscribe
        async def handle_published(event: TrackPublishedEvent):
            received_events.append(event)

        # Send raw protobuf message (auto-wrapped by manager)
        manager.send(proto_published)
        await manager.wait()

        assert len(received_events) == 1
        assert received_events[0].user_id == "user123"
        # Verify participant property returns correct value from protobuf payload
        assert received_events[0].participant is not None
        assert received_events[0].participant.user_id == "user123"
        assert received_events[0].participant.session_id == "session456"
        assert hasattr(received_events[0], "event_id")

        # Test TrackUnpublishedEvent
        proto_unpublished = events_pb2.TrackUnpublished(
            user_id="user456", participant=participant, cause=1
        )

        unpublished_event = TrackUnpublishedEvent.from_proto(proto_unpublished)
        assert unpublished_event.user_id == "user456"
        assert unpublished_event.participant is not None
        assert unpublished_event.participant.user_id == "user123"
        assert unpublished_event.cause == 1
