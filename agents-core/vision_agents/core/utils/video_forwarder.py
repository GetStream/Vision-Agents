import asyncio
import datetime
import logging
from typing import Optional, Callable, Any

import av
from aiortc import VideoStreamTrack
from av.frame import Frame

from vision_agents.core.utils.video_queue import VideoLatestNQueue

logger = logging.getLogger(__name__)

class VideoForwarder:
    """
    Pulls frames from `input_track` into a latest-N buffer.
    Consumers can:
      - call `await next_frame()` (pull model), OR
      - register callbacks with `add_frame_handler()` and start with `start_consumers()` (push model).
    `fps` limits how often frames are forwarded to consumers (coalescing to newest).
    """
    def __init__(self, input_track: VideoStreamTrack, *, max_buffer: int = 10, fps: Optional[float] = 30, name: str = "video-forwarder"):
        self.input_track = input_track
        self.queue: VideoLatestNQueue[Frame] = VideoLatestNQueue(maxlen=max_buffer)
        self.fps = fps  # None = unlimited, else forward at ~fps
        self._tasks: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()
        self._started = False
        self.name = name
        self._frame_handlers: list[tuple[Callable[[av.VideoFrame], Any], dict[str, Any]]] = []

    # ---------- lifecycle ----------
    async def start(self) -> None:
        if self._started:
            logger.warning("%s: start() called but already started", self.name)
            return
        self._started = True
        self._stopped.clear()
        task = asyncio.create_task(self._producer())
        task.add_done_callback(self._task_done)
        self._tasks.add(task)

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopped.set()
        self._started = False
        # Create snapshot of tasks to avoid race conditions
        tasks_snapshot = list(self._tasks)
        for t in tasks_snapshot:
            t.cancel()
        if tasks_snapshot:
            await asyncio.gather(*tasks_snapshot, return_exceptions=True)
        self._tasks.clear()
        # drain queue
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    def _task_done(self, task: asyncio.Task) -> None:
        """Callback to remove completed tasks from the set."""
        self._tasks.discard(task)
        exc = task.exception()
        if exc:
            logger.error("%s: Task failed with exception: %s", self.name, exc, exc_info=exc)

        if task.cancelled():
            return

    # ---------- producer (fills latest-N buffer) ----------
    async def _producer(self):
        try:
            while not self._stopped.is_set():
                frame : Frame = await self.input_track.recv()
                frame.dts = int(datetime.datetime.now().timestamp())

                await self.queue.put_latest(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("%s: Producer failed with exception: %s", self.name, e, exc_info=True)
            raise

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

    # ---------- frame handler management ----------
    def add_frame_handler(
        self,
        on_frame: Callable[[av.VideoFrame], Any],
        *,
        fps: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a callback to be called for each frame.
        
        Args:
            on_frame: Callback function (sync or async) to receive frames
            fps: Frame rate for this handler (overrides default). None = unlimited.
            name: Optional name for this handler (for logging)
        """
        handler_name = name or f"handler-{len(self._frame_handlers)}"
        config = {
            'fps': fps if fps is not None else self.fps,
            'name': handler_name,
        }
        self._frame_handlers.append((on_frame, config))
        logger.debug("%s: Added frame handler '%s' with fps=%s", self.name, handler_name, config['fps'])
    
    def remove_frame_handler(self, on_frame: Callable[[av.VideoFrame], Any]) -> bool:
        """
        Remove a previously registered callback.
        
        Args:
            on_frame: The callback to remove
            
        Returns:
            True if the handler was found and removed, False otherwise
        """
        original_len = len(self._frame_handlers)
        self._frame_handlers = [(cb, cfg) for cb, cfg in self._frame_handlers if cb != on_frame]
        removed = len(self._frame_handlers) < original_len
        if removed:
            logger.debug("%s: Removed frame handler", self.name)
        return removed
    
    async def start_consumers(self) -> None:
        """
        Start consumer tasks for all registered frame handlers.
        Each handler runs in its own task with its own fps throttling.
        """
        for on_frame, config in self._frame_handlers:
            await self._start_consumer_task(
                on_frame=on_frame,
                fps=config['fps'],
                consumer_name=config['name'],
            )
        logger.info("%s: Started %d consumer task(s)", self.name, len(self._frame_handlers))

    # ---------- push model (broadcast via callback) ----------
    async def _start_consumer_task(
        self,
        on_frame: Callable[[av.VideoFrame], Any],
        *,
        fps: Optional[float] = None,
        log_interval_seconds: float = 10.0,
        consumer_name: Optional[str] = None,
    ) -> None:
        """
        Internal method to start a single consumer task.
        
        Args:
            on_frame: Callback function to receive frames
            fps: Frame rate for this consumer. None = unlimited.
            log_interval_seconds: How often to log consumer statistics
            consumer_name: Optional name for this consumer (for logging)
        """
        consumer_fps = fps if fps is not None else self.fps
        consumer_label = consumer_name or "consumer"
        
        async def _consumer():
            loop = asyncio.get_running_loop()
            min_interval = (1.0 / consumer_fps) if (consumer_fps and consumer_fps > 0) else 0.0
            last_ts = 0.0
            is_coro = asyncio.iscoroutinefunction(on_frame)
            frames_forwarded = 0
            last_log = loop.time()
            last_width: Optional[int] = None
            last_height: Optional[int] = None
            while not self._stopped.is_set():
                # Wait for at least one frame
                frame = await self.next_frame()
                # track latest resolution for summary logs
                try:
                    last_width = int(getattr(frame, "width", 0)) or last_width
                    last_height = int(getattr(frame, "height", 0)) or last_height
                except Exception:
                    # ignore resolution extraction errors
                    pass
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
                frames_forwarded += 1
                # periodic summary logging
                if log_interval_seconds > 0:
                    now_time = loop.time()
                    if (now_time - last_log) >= log_interval_seconds:
                        if last_width and last_height:
                            logger.info(
                                "%s [%s] forwarded %d frames at %dx%d resolution in the last %.0f seconds (target: %.1f fps)",
                                self.name,
                                consumer_label,
                                frames_forwarded,
                                last_width,
                                last_height,
                                log_interval_seconds,
                                consumer_fps or 0,
                            )
                        else:
                            logger.info(
                                "%s [%s] forwarded %d frames in the last %.0f seconds (target: %.1f fps)",
                                self.name,
                                consumer_label,
                                frames_forwarded,
                                log_interval_seconds,
                                consumer_fps or 0,
                            )
                        frames_forwarded = 0
                        last_log = now_time

        task = asyncio.create_task(_consumer())
        task.add_done_callback(self._task_done)
        self._tasks.add(task)

    async def start_event_consumer(
        self,
        on_frame: Callable[[av.VideoFrame], Any],  # async or sync
        *,
        fps: Optional[float] = None,
        log_interval_seconds: float = 10.0,
        consumer_name: Optional[str] = None,
    ) -> None:
        """
        Starts a task that calls `on_frame(latest_frame)` at ~fps.
        If fps is None, it forwards as fast as frames arrive (still coalescing).
        
        Deprecated: Consider using `add_frame_handler()` + `start_consumers()` instead
        for better support of multiple handlers.
        
        Args:
            on_frame: Callback function to receive frames
            fps: Frame rate for this consumer (overrides default). None = unlimited.
            log_interval_seconds: How often to log consumer statistics
            consumer_name: Optional name for this consumer (for logging)
        """
        await self._start_consumer_task(
            on_frame=on_frame,
            fps=fps,
            log_interval_seconds=log_interval_seconds,
            consumer_name=consumer_name,
        )
