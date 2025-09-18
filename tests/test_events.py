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

    await manager.send(ValidEvent(field=2))
    assert my_handler.value == 2

@pytest.mark.asyncio
async def test_subscribe_with_multiple_events_different():
    manager = EventManager()
    manager.register(ValidEvent)
    manager.register(AnotherEvent)

    with pytest.raises(RuntimeError):
        @manager.subscribe
        async def multi_event_handler(event1: ValidEvent, event2: AnotherEvent):
            value += 1


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

    await manager.send(ValidEvent(field=1))
    await manager.send(AnotherEvent(value=2))

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

    await manager.send(ValidEvent(field=10))

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
        await manager.send(UnregisteredEvent(data="oops"))
