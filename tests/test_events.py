import pytest
import types
import dataclasses

from vision_agents.core.events.manager import EventManager, ExceptionEvent


@dataclasses.dataclass
class InvalidEvent:
    # Missing `type` attribute and name does not end with 'Event'
    field: int


@dataclasses.dataclass
class ValidEvent:
    field: int
    type: str = "custom.validevent"


@dataclasses.dataclass
class AnotherEvent:
    value: str
    type: str = "custom.anotherevent"


@pytest.mark.asyncio
async def test_register_invalid_event_raises_value_error():
    manager = EventManager()
    with pytest.raises(ValueError):
        manager.register(InvalidEvent)


@pytest.mark.asyncio
async def test_register_valid_event_success():
    manager = EventManager()
    manager.register(ValidEvent)
    # after registration the event type should be in the internal dict
    assert "custom.validevent" in manager._events


@pytest.mark.asyncio
async def test_register_events_from_module_raises_name_error():
    manager = EventManager()

    # Create a dummy module with two event classes
    dummy_module = types.SimpleNamespace(
        MyEvent=ValidEvent,
        Another=AnotherEvent,
    )
    dummy_module.__name__ = "dummy_module"
    manager.register_events_from_module(dummy_module, prefix="custom.")

    @manager.subscribe
    async def my_handler(event: ValidEvent):
        my_handler.value = event.field

    manager.send(ValidEvent(field=2))
    await manager.wait()
    assert my_handler.value == 2


@pytest.mark.asyncio
async def test_subscribe_with_multiple_events_different():
    manager = EventManager()
    manager.register(ValidEvent)
    manager.register(AnotherEvent)

    with pytest.raises(RuntimeError):

        @manager.subscribe
        async def multi_event_handler(event1: ValidEvent, event2: AnotherEvent):
            pass


@pytest.mark.asyncio
async def test_subscribe_with_multiple_events_as_one_processes():
    manager = EventManager()
    manager.register(ValidEvent)
    manager.register(AnotherEvent)
    value = 0

    @manager.subscribe
    async def multi_event_handler(event: ValidEvent | AnotherEvent):
        nonlocal value
        value += 1

    manager.send(ValidEvent(field=1))
    manager.send(AnotherEvent(value="2"))
    await manager.wait()

    assert value == 2


@pytest.mark.asyncio
async def test_subscribe_unregistered_event_raises_key_error():
    manager = EventManager(ignore_unknown_events=False)

    with pytest.raises(KeyError):

        @manager.subscribe
        async def unknown_handler(event: ValidEvent):
            pass


@pytest.mark.asyncio
async def test_handler_exception_triggers_recursive_exception_event():
    manager = EventManager()
    manager.register(ValidEvent, ignore_not_compatible=False)
    manager.register(ExceptionEvent)

    # Counter to ensure recursive handler is invoked
    recursive_counter = {"count": 0}

    @manager.subscribe
    async def failing_handler(event: ValidEvent):
        raise RuntimeError("Intentional failure")

    @manager.subscribe
    async def exception_handler(event: ExceptionEvent):
        # Increment the counter each time the exception handler runs
        recursive_counter["count"] += 1
        # Re-raise the exception only once to trigger a second recursion
        if recursive_counter["count"] == 1:
            raise ValueError("Re-raising in exception handler")

    manager.send(ValidEvent(field=10))
    await manager.wait()

    # After processing, the recursive counter should be 2 (original failure + one re-raise)
    assert recursive_counter["count"] == 2


@pytest.mark.asyncio
async def test_send_unknown_event_type_raises_key_error():
    manager = EventManager(ignore_unknown_events=False)

    # Define a dynamic event class that is not registered
    @dataclasses.dataclass
    class UnregisteredEvent:
        data: str
        type: str = "custom.unregistered"

    # The event will be queued but there are no handlers for its type
    with pytest.raises(RuntimeError):
        manager.send(UnregisteredEvent(data="oops"))


@pytest.mark.asyncio
async def test_merge_managers_events_processed_in_one():
    """Test that when two managers are merged, events from both are processed in the merged manager."""
    # Create two separate managers
    manager1 = EventManager()
    manager2 = EventManager()

    # Register different events in each manager
    manager1.register(ValidEvent)
    manager2.register(AnotherEvent)

    # Set up handlers in each manager
    all_events_processed: list[tuple[str, ValidEvent | AnotherEvent]] = []

    @manager1.subscribe
    async def manager1_handler(event: ValidEvent):
        all_events_processed.append(("manager1", event))

    @manager2.subscribe
    async def manager2_handler(event: AnotherEvent):
        all_events_processed.append(("manager2", event))

    # Send events to both managers before merging
    manager1.send(ValidEvent(field=1))
    manager2.send(AnotherEvent(value="test"))

    # Wait for events to be processed in their respective managers
    await manager1.wait()
    await manager2.wait()

    # Verify events were processed in their original managers
    assert len(all_events_processed) == 2
    assert all_events_processed[0][0] == "manager1"
    assert isinstance(all_events_processed[0][1], ValidEvent)
    assert all_events_processed[0][1].field == 1
    assert all_events_processed[1][0] == "manager2"
    assert isinstance(all_events_processed[1][1], AnotherEvent)
    assert all_events_processed[1][1].value == "test"

    # Clear the processed events list
    all_events_processed.clear()

    # Merge manager2 into manager1
    manager1.merge(manager2)

    # Verify that manager2's processing task is stopped
    assert manager2._processing_task is None

    # Send new events to both managers after merging
    manager1.send(ValidEvent(field=2))
    manager2.send(AnotherEvent(value="merged"))

    # Wait for events to be processed (only manager1's task should be running)
    await manager1.wait()

    # After merging, both events should be processed by manager1's task
    # (manager2's processing task should be stopped)
    assert len(all_events_processed) == 2
    # Both events should be processed by manager1's task
    assert all_events_processed[0][0] == "manager1"  # ValidEvent
    assert isinstance(all_events_processed[0][1], ValidEvent)
    assert all_events_processed[0][1].field == 2
    assert (
        all_events_processed[1][0] == "manager2"
    )  # AnotherEvent (handler from manager2)
    assert isinstance(all_events_processed[1][1], AnotherEvent)
    assert all_events_processed[1][1].value == "merged"

    # Verify that manager2 can still send events but they go to manager1's queue
    # and are processed by manager1's task
    all_events_processed.clear()
    manager2.send(AnotherEvent(value="from_manager2"))
    await manager1.wait()

    # The event from manager2 should be processed by manager1's task
    assert len(all_events_processed) == 1
    assert all_events_processed[0][0] == "manager2"  # Handler from manager2
    assert isinstance(all_events_processed[0][1], AnotherEvent)
    assert all_events_processed[0][1].value == "from_manager2"


@pytest.mark.asyncio
async def test_merge_managers_preserves_silent_events(caplog):
    """Test that when two managers are merged, silent events from both are preserved."""
    import logging

    manager1 = EventManager()
    manager2 = EventManager()

    manager1.register(ValidEvent)
    manager2.register(AnotherEvent)

    # Mark ValidEvent as silent in manager1
    manager1.silent(ValidEvent)
    # Mark AnotherEvent as silent in manager2
    manager2.silent(AnotherEvent)

    handler_called = []

    @manager1.subscribe
    async def valid_handler(event: ValidEvent):
        handler_called.append("valid")

    @manager2.subscribe
    async def another_handler(event: AnotherEvent):
        handler_called.append("another")

    # Merge manager2 into manager1
    manager1.merge(manager2)

    # Verify that both silent events are preserved
    assert "custom.validevent" in manager1._silent_events
    assert "custom.anotherevent" in manager1._silent_events

    # Verify that manager2 also references the merged silent events
    assert manager2._silent_events is manager1._silent_events

    # Capture logs at INFO level
    with caplog.at_level(logging.INFO):
        # Send both events
        manager1.send(ValidEvent(field=42))
        manager1.send(AnotherEvent(value="test"))
        await manager1.wait()

    # Both handlers should have been called
    assert handler_called == ["valid", "another"]

    # Check log messages
    log_messages = [record.message for record in caplog.records]

    # Should NOT see "Called handler" for either event (both are silent)
    assert not any("Called handler valid_handler" in msg for msg in log_messages)
    assert not any("Called handler another_handler" in msg for msg in log_messages)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_protobuf_events_with_base_event():
    """Test that event manager handles protobuf events that inherit from BaseEvent."""
    from vision_agents.core.events.manager import EventManager
    from vision_agents.core.edge.sfu_events import (
        AudioLevelEvent,
        ParticipantJoinedEvent,
    )
    from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
    from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

    manager = EventManager()

    # Register generated protobuf event classes
    manager.register(AudioLevelEvent)
    manager.register(ParticipantJoinedEvent)

    assert AudioLevelEvent.type in manager._events
    assert ParticipantJoinedEvent.type in manager._events

    # Test 1: Send wrapped protobuf event with BaseEvent fields
    proto_audio = events_pb2.AudioLevel(user_id="user123", level=0.85, is_speaking=True)
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
    proto_raw = events_pb2.AudioLevel(user_id="user456", level=0.95, is_speaking=False)

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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_track_published_event_with_participant_property():
    """Test that TrackPublishedEvent correctly handles participant property override."""
    from vision_agents.core.events.manager import EventManager
    from vision_agents.core.edge.sfu_events import (
        TrackPublishedEvent,
        TrackUnpublishedEvent,
    )
    from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
    from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

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
