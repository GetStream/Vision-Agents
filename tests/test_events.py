import pytest
import types
import dataclasses

from stream_agents.core.events.manager import EventManager, ExceptionEvent


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
    manager.send(AnotherEvent(value=2))
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
    all_events_processed = []
    
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
    assert all_events_processed[0][1].field == 1
    assert all_events_processed[1][0] == "manager2"
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
    assert all_events_processed[0][1].field == 2
    assert all_events_processed[1][0] == "manager2"  # AnotherEvent (handler from manager2)
    assert all_events_processed[1][1].value == "merged"
    
    # Verify that manager2 can still send events but they go to manager1's queue
    # and are processed by manager1's task
    all_events_processed.clear()
    manager2.send(AnotherEvent(value="from_manager2"))
    await manager1.wait()
    
    # The event from manager2 should be processed by manager1's task
    assert len(all_events_processed) == 1
    assert all_events_processed[0][0] == "manager2"  # Handler from manager2
    assert all_events_processed[0][1].value == "from_manager2"
