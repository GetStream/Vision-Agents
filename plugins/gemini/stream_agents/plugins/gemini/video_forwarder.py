import asyncio
from typing import Optional, Callable, Any

import av
from aiortc import VideoStreamTrack

from stream_agents.plugins.gemini.queue import LatestNQueue


class VideoForwarder:
    """
    Pulls frames from `input_track` into a latest-N buffer.
    Consumers can:
      - call `await next_frame()` (pull model), OR
      - run `start_event_consumer(on_frame)` (push model via callback).
    `fps` limits how often frames are forwarded to consumers (coalescing to newest).
    """
    def __init__(self, input_track: VideoStreamTrack, *, max_buffer: int = 10, fps: Optional[float] = 30):
        self.input_track = input_track
        self.queue: LatestNQueue[av.VideoFrame] = LatestNQueue(maxlen=max_buffer)
        self.fps = fps  # None = unlimited, else forward at ~fps
        self._tasks: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()

    # ---------- lifecycle ----------
    async def start(self) -> None:
        self._stopped.clear()
        self._tasks.add(asyncio.create_task(self._producer()))

    async def stop(self) -> None:
        self._stopped.set()
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        # drain queue
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    # ---------- producer (fills latest-N buffer) ----------
    async def _producer(self):
        try:
            while not self._stopped.is_set():
                frame = await self.input_track.recv()
                await self.queue.put_latest(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            # optional: log
            pass

    # ---------- consumer API (pull one frame; coalesce backlog to newest) ----------
    async def next_frame(self, *, timeout: Optional[float] = None) -> av.VideoFrame:
        """
        Returns the newest available frame. If there's backlog, older frames
        are drained so you get the latest (low latency).
        """
        if timeout is None:
            frame = await self.queue.get()
        else:
            async with asyncio.timeout(timeout):
                frame = await self.queue.get()

        # drain to newest
        while True:
            try:
                newer = self.queue.get_nowait()
                frame = newer
            except asyncio.QueueEmpty:
                break
        return frame

    # ---------- push model (broadcast via callback) ----------
    async def start_event_consumer(
        self,
        on_frame: Callable[[av.VideoFrame], Any],  # async or sync
    ) -> None:
        """
        Starts a task that calls `on_frame(latest_frame)` at ~fps.
        If fps is None, it forwards as fast as frames arrive (still coalescing).
        """
        async def _consumer():
            loop = asyncio.get_running_loop()
            min_interval = (1.0 / self.fps) if (self.fps and self.fps > 0) else 0.0
            last_ts = 0.0
            is_coro = asyncio.iscoroutinefunction(on_frame)
            try:
                while not self._stopped.is_set():
                    # Wait for at least one frame
                    frame = await self.next_frame()
                    # Throttle to fps (if set)
                    if min_interval > 0.0:
                        now = loop.time()
                        elapsed = now - last_ts
                        if elapsed < min_interval:
                            # coalesce: keep draining to newest until it's time
                            await asyncio.sleep(min_interval - elapsed)
                        last_ts = loop.time()
                    # Call handler
                    if is_coro:
                        await on_frame(frame)  # type: ignore[arg-type]
                    else:
                        on_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                # optional: log
                pass

        self._tasks.add(asyncio.create_task(_consumer()))
