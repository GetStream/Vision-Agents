import asyncio

import pytest
from vision_agents.core.utils.stream import (
    Stream,
    StreamClosed,
    StreamEmpty,
    StreamFull,
)


@pytest.fixture
def stream() -> Stream[int]:
    return Stream[int]()


@pytest.fixture
def bounded() -> Stream[int]:
    return Stream[int](maxsize=2)


class TestStream:
    async def test_new_stream_is_empty_and_open(self, stream: Stream[int]):
        assert stream.empty()
        assert not stream.closed()
        assert not stream.full()
        assert stream.size() == 0

    async def test_unbounded_stream_is_never_full(self, stream: Stream[int]):
        for i in range(100):
            stream.send_nowait(i)
        assert not stream.full()
        assert stream.size() == 100

    async def test_bounded_stream_reports_full(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)
        assert bounded.full()
        assert bounded.size() == 2

    async def test_negative_maxsize_treated_as_unbounded(self):
        s: Stream[int] = Stream(maxsize=-5)
        for i in range(50):
            s.send_nowait(i)
        assert not s.full()

    async def test_send_and_get_nowait_fifo(self, stream: Stream[int]):
        stream.send_nowait(1)
        stream.send_nowait(2)
        stream.send_nowait(3)
        assert stream.get_nowait() == 1
        assert stream.get_nowait() == 2
        assert stream.get_nowait() == 3

    async def test_send_nowait_on_closed_raises(self, stream: Stream[int]):
        stream.close()
        with pytest.raises(StreamClosed):
            stream.send_nowait(1)

    async def test_send_nowait_on_full_raises(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)
        with pytest.raises(StreamFull):
            bounded.send_nowait(3)

    async def test_get_nowait_empty_raises(self, stream: Stream[int]):
        with pytest.raises(StreamEmpty):
            stream.get_nowait()

    async def test_get_nowait_closed_empty_raises_closed(self, stream: Stream[int]):
        stream.close()
        with pytest.raises(StreamClosed):
            stream.get_nowait()

    async def test_send_get_roundtrip(self, stream: Stream[int]):
        await stream.send(42)
        assert await stream.get() == 42

    async def test_get_blocks_until_send(self, stream: Stream[int]):
        result: list[int] = []

        async def consumer():
            result.append(await stream.get())

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)
        assert not task.done()

        await stream.send(99)
        await task
        assert result == [99]

    async def test_send_blocks_when_full(self, bounded: Stream[int]):
        await bounded.send(1)
        await bounded.send(2)

        send_done = asyncio.Event()

        async def producer():
            await bounded.send(3)
            send_done.set()

        task = asyncio.create_task(producer())
        await asyncio.sleep(0)
        assert not send_done.is_set()

        assert await bounded.get() == 1
        await task
        assert send_done.is_set()
        assert await bounded.get() == 2
        assert await bounded.get() == 3

    async def test_close_sets_closed(self, stream: Stream[int]):
        stream.close()
        assert stream.closed()

    async def test_send_on_closed_raises(self, stream: Stream[int]):
        stream.close()
        with pytest.raises(StreamClosed):
            await stream.send(1)

    async def test_get_on_closed_empty_raises(self, stream: Stream[int]):
        stream.close()
        with pytest.raises(StreamClosed):
            await stream.get()

    async def test_get_drains_before_raising_closed(self, stream: Stream[int]):
        await stream.send(1)
        await stream.send(2)
        stream.close()
        assert await stream.get() == 1
        assert await stream.get() == 2
        with pytest.raises(StreamClosed):
            await stream.get()

    async def test_get_nowait_drains_before_raising_closed(self, stream: Stream[int]):
        await stream.send(1)
        await stream.send(2)
        stream.close()
        assert stream.get_nowait() == 1
        assert stream.get_nowait() == 2
        with pytest.raises(StreamClosed):
            stream.get_nowait()

    async def test_closed_sender_raises_after_clear(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)

        async def producer():
            await bounded.send(3)

        task = asyncio.create_task(producer())
        await asyncio.sleep(0)
        bounded.close()
        bounded.clear()
        with pytest.raises(StreamClosed):
            await task

    async def test_async_for_yields_items_then_stops(self, stream: Stream[int]):
        items = [10, 20, 30]
        for item in items:
            await stream.send(item)
        stream.close()

        collected: list[int] = []
        async for value in stream:
            collected.append(value)
        assert collected == items

    async def test_close_exits_running_iterator(self, stream: Stream[int]):
        collected: list[int] = []

        async def consumer():
            async for value in stream:
                collected.append(value)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)

        await stream.send(1)
        await stream.send(2)
        await asyncio.sleep(0)
        stream.close()
        await task
        assert collected == [1, 2]

    async def test_clear_empties_stream(self, stream: Stream[int]):
        await stream.send(1)
        await stream.send(2)
        stream.clear()
        assert stream.empty()
        assert stream.size() == 0

    async def test_clear_unblocks_sender(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)

        send_done = asyncio.Event()

        async def producer():
            await bounded.send(3)
            send_done.set()

        task = asyncio.create_task(producer())
        await asyncio.sleep(0)
        assert not send_done.is_set()

        bounded.clear()
        await task
        assert send_done.is_set()

    async def test_clear_unblocks_all_blocked_senders(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)

        async def producer(value: int):
            await bounded.send(value)

        tasks = [asyncio.create_task(producer(v)) for v in (3, 4, 5)]
        await asyncio.sleep(0)
        assert all(not t.done() for t in tasks)

        bounded.clear()

        # clear() frees maxsize=2 slots, so only the first two senders wake
        # and fill the buffer.
        await asyncio.wait_for(tasks[0], timeout=1.0)
        await asyncio.wait_for(tasks[1], timeout=1.0)
        # The third sender stays parked — no capacity left until a get() runs.
        assert not tasks[2].done()

        assert await bounded.get() == 3
        assert await bounded.get() == 4

        # Draining freed a slot, which wakes the third sender.
        await asyncio.wait_for(tasks[2], timeout=1.0)
        assert await bounded.get() == 5

    async def test_clear_keeps_iterator_running(self, stream: Stream[int]):
        collected: list[int] = []

        async def consumer():
            async for value in stream:
                collected.append(value)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)

        await stream.send(1)
        await asyncio.sleep(0)
        await stream.send(10)
        await stream.send(20)
        stream.clear()
        await stream.send(2)
        await asyncio.sleep(0)
        stream.close()
        await task
        assert collected == [1, 2]

    async def test_cancelled_getter_propagates(self, stream: Stream[int]):
        task = asyncio.create_task(stream.get())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancelled_sender_propagates(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)
        task = asyncio.create_task(bounded.send(3))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancelled_getter_does_not_lose_item(self, stream: Stream[int]):
        task = asyncio.create_task(stream.get())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await stream.send(42)
        assert await stream.get() == 42

    async def test_cancelled_sender_does_not_corrupt_stream(self, bounded: Stream[int]):
        bounded.send_nowait(1)
        bounded.send_nowait(2)

        task = asyncio.create_task(bounded.send(3))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await bounded.get() == 1
        await bounded.send(4)
        assert await bounded.get() == 2
        assert await bounded.get() == 4

    async def test_multiple_getters_served_in_order(self, stream: Stream[int]):
        results: list[int] = []

        async def getter(idx: int):
            val = await stream.get()
            results.append(val)

        tasks = [asyncio.create_task(getter(i)) for i in range(3)]
        await asyncio.sleep(0)

        await stream.send(10)
        await stream.send(20)
        await stream.send(30)
        await asyncio.gather(*tasks)
        assert results == [10, 20, 30]

    async def test_multiple_senders_served_in_order(self):
        s: Stream[int] = Stream(maxsize=1)
        s.send_nowait(0)

        sent: list[int] = []

        async def sender(val: int):
            await s.send(val)
            sent.append(val)

        tasks = [asyncio.create_task(sender(i)) for i in range(1, 4)]
        await asyncio.sleep(0)

        for _ in range(4):
            await s.get()
            await asyncio.sleep(0)

        await asyncio.gather(*tasks)
        assert sent == [1, 2, 3]

    async def test_repr(self, stream: Stream[int]):
        assert "Stream" in repr(stream)
        assert "closed=False" in repr(stream)
        stream.close()
        assert "closed=True" in repr(stream)

    async def test_str(self, stream: Stream[int]):
        assert "Stream" in str(stream)
        assert "closed=False" in str(stream)
        stream.close()
        assert "closed=True" in str(stream)

    async def test_peek_snapshots_buffered_items_without_draining(
        self, stream: Stream[int]
    ):
        assert stream.peek() == []

        await stream.send(1)
        await stream.send(2)
        await stream.send(3)

        assert stream.peek() == [1, 2, 3]
        assert stream.peek() == [1, 2, 3]

        assert await stream.get() == 1
        assert stream.peek() == [2, 3]

    async def test_peek_returns_independent_copy(self, stream: Stream[int]):
        await stream.send(1)
        await stream.send(2)

        snapshot = stream.peek()
        snapshot.append(999)
        snapshot.clear()

        assert stream.peek() == [1, 2]

    async def test_collect_none_drains_until_closed(self, stream: Stream[int]):
        await stream.send(1)
        await stream.send(2)
        stream.close()

        assert await stream.collect(None) == [1, 2]
        assert stream.empty()

    async def test_collect_default_waits_until_closed_like_none(
        self, stream: Stream[int]
    ):
        await stream.send(7)
        stream.close()

        assert await stream.collect() == [7]

    async def test_collect_with_timeout_returns_buffered_then_stops(
        self, stream: Stream[int]
    ):
        stream.send_nowait(1)

        assert await stream.collect(timeout=0.1) == [1]
        assert stream.empty()
        assert not stream.closed()

    async def test_collect_timeout_zero_on_empty_returns_empty(
        self, stream: Stream[int]
    ):
        assert await stream.collect(timeout=0) == []

    async def test_collect_timeout_zero_drains_all_buffered_without_waiting(
        self, stream: Stream[int]
    ):
        for i in range(3):
            stream.send_nowait(i)

        assert await stream.collect(timeout=0) == [0, 1, 2]

    async def test_collect_negative_timeout_raises(self, stream: Stream[int]):
        with pytest.raises(ValueError, match="timeout"):
            await stream.collect(timeout=-1.0)
