import asyncio
from typing import Generic, TypeVar

T = TypeVar("T")


class VideoLatestNQueue(asyncio.Queue, Generic[T]):
    """
    A generic asyncio queue that always keeps only the latest N items.
    If full on put, it discards oldest items to make room.
    """

    def __init__(self, maxlen: int):
        super().__init__(maxsize=maxlen)

    async def put_latest(self, item: T) -> None:
        while self.full():
            try:
                self.get_nowait()
            except asyncio.QueueEmpty:
                break
        await super().put(item)

    def put_latest_nowait(self, item: T) -> None:
        while self.full():
            try:
                self.get_nowait()
            except asyncio.QueueEmpty:
                break
        super().put_nowait(item)
