import asyncio
import contextlib
from collections import deque
from typing import AsyncIterator, Generic, TypeVar

T = TypeVar("T")


class StreamClosed(Exception): ...


class StreamEmpty(Exception): ...


class StreamFull(Exception): ...


class Stream(Generic[T]):
    """
    A generic iterator abstraction to pass data between components: pcm, chunks, etc.
    Can be closed or cleared.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._getters = deque[asyncio.Future]()
        self._senders = deque[asyncio.Future]()
        self._items = deque[T]()
        self._closed = asyncio.Event()
        self._maxsize = max(maxsize, 0)

    async def __anext__(self) -> T:
        try:
            return await self.get()
        except StreamClosed:
            raise StopAsyncIteration

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    def close(self) -> None:
        """
        Close the queue allowing iterators to exit
        """
        self._closed.set()
        while self._getters:
            self._wakeup_next_getter()
        while self._senders:
            self._wakeup_next_sender()

    def closed(self) -> bool:
        return self._closed.is_set()

    def clear(self) -> None:
        """
        Empty the stream but allow the running iterators to keep going.
        """
        self._items.clear()
        self._wakeup_next_sender()

    def empty(self) -> bool:
        return not self._items

    def size(self) -> int:
        return len(self._items)

    def full(self) -> bool:
        return 0 < self._maxsize <= self.size()

    async def send(self, item: T) -> None:
        """Put an item into the queue.

        Put an item into the queue. If the queue is full, wait until a free
        slot is available before adding item.

        Raises QueueShutDown if the queue has been shut down.
        """
        while self.full():
            if self.closed():
                raise StreamClosed("Stream is closed")
            sender = asyncio.Future()
            self._senders.append(sender)
            try:
                await sender
            except:
                sender.cancel()
                with contextlib.suppress(ValueError):
                    self._senders.remove(sender)
                if not self.full() and not sender.cancelled():
                    self._wakeup_next_sender()
                raise
        return self.send_nowait(item)

    def send_nowait(self, item: T) -> None:
        if self.closed():
            raise StreamClosed("Stream is closed")
        if self.full():
            raise StreamFull("Stream is full")
        self._items.append(item)
        self._wakeup_next_getter()

    async def get(self) -> T:
        while self.empty():
            if self.closed() and self.empty():
                raise StreamClosed("Stream is closed")
            getter = asyncio.Future()
            self._getters.append(getter)
            try:
                await getter
            except:
                getter.cancel()  # Just in case getter is not done yet.
                with contextlib.suppress(ValueError):
                    # Clean self._getters from canceled getters.
                    self._getters.remove(getter)
                if not self.empty() and not getter.cancelled():
                    self._wakeup_next_getter()
                raise
        return self.get_nowait()

    def get_nowait(self) -> T:
        if not self.empty():
            item = self._items.popleft()
            self._wakeup_next_sender()
            return item
        elif self.closed():
            raise StreamClosed("Stream is closed")
        else:
            raise StreamEmpty("Stream is empty")

    def peek(self) -> list[T]:
        """
        A helper method to peek into already buffered items without emptying the buffer.

        Returns:
            list of items from the stream.
        """
        return list(self._items)

    def _wakeup_next_sender(self):
        # Notify the next waiting "send()" call that
        # there's a free space in the items queue
        self._wakeup_next(self._senders)

    def _wakeup_next_getter(self):
        # Notify the next waiting "get()" call that there's new data available
        self._wakeup_next(self._getters)

    def _wakeup_next(self, waiters: deque[asyncio.Future]) -> None:
        # Go over the waiters and resolve the next waiting in line
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                break

    def __repr__(self):
        return f"<{type(self).__name__} closed={self.closed()}>"

    def __str__(self):
        return f"<{type(self).__name__} closed={self.closed()}>"
