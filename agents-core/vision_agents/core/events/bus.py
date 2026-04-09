import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")
EventHandler = Callable[[T], Awaitable[None]]


class EventBus(Protocol[T]):
    def subscribe(self, handler: EventHandler[T]) -> None: ...

    async def publish(self, event: T) -> None: ...


class InMemoryEventBus(Generic[T]):
    def __init__(self) -> None:
        self._handlers: list[EventHandler[T]] = []

    def subscribe(self, handler: EventHandler[T]) -> None:
        if not inspect.iscoroutinefunction(handler):
            raise RuntimeError(
                "Handlers must be coroutines. Use async def handler(event):"
            )
        self._handlers.append(handler)

    async def publish(self, event: T) -> None:
        if not self._handlers:
            return
        await asyncio.gather(*(handler(event) for handler in list(self._handlers)))
