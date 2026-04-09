import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class EventBus(Protocol[T]):
    def subscribe(self, handler: Callable[[T], Awaitable[None]]) -> None: ...

    async def publish(self, event: T) -> None: ...


class InMemoryEventBus(Generic[T]):
    def __init__(self) -> None:
        self._handlers: list[Callable[[T], Awaitable[None]]] = []

    def subscribe(self, handler: Callable[[T], Awaitable[None]]) -> None:
        if not asyncio.iscoroutinefunction(handler):
            raise RuntimeError(
                "Handlers must be coroutines. Use async def handler(event):"
            )
        self._handlers.append(handler)

    async def publish(self, event: T) -> None:
        if not self._handlers:
            return
        await asyncio.gather(*(handler(event) for handler in list(self._handlers)))
