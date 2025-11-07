import asyncio
import datetime
import logging
from typing import Optional, Callable, Any

import av
from aiortc import VideoStreamTrack
from av.frame import Frame
from hatch.cli import self

from vision_agents.core.utils.video_queue import VideoLatestNQueue

logger = logging.getLogger(__name__)

class VideoForwarder:
    """
    VideoForwarder handles forwarding a video track to 1 or multiple targets

    Example:

        forwarder = VideoForwarder(input_track=track, fps=5)
        forwarder.add_frame_handler( lamba x: print("received frame"), fps =1 )
        forwarder.stop()

        # start's automatically when attaching handlers

    """
    _producer_task: Optional[asyncio.Task] = None
    _consumer_task: Optional[asyncio.Task] = None
    _frame_handlers: list[tuple[Callable[[av.VideoFrame], Any], dict[str, Any]]] = []
    _started = False

    def __init__(self, input_track: VideoStreamTrack, *, max_buffer: int = 10, fps: Optional[float] = 30, name: str = "video-forwarder"):
        self.name = name
        self.input_track = input_track
        self.queue: VideoLatestNQueue[Frame] = VideoLatestNQueue(maxlen=max_buffer)
        self.fps = fps  # None = unlimited, else forward at ~fps

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
        if fps > self.fps:
            raise ValueError("fps on handler %d cannot be greater than fps on forwarder %d" % (fps, self.fps))
        config = {
            'fps': fps if fps is not None else self.fps,
            'name': handler_name,
        }

        self._frame_handlers.append((on_frame, config))
        self.start()

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

        if len(self._frame_handlers) == 0:
            self.stop()
        return removed

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._producer_task = asyncio.create_task(self._producer())
        self._consumer_task = asyncio.create_task(self._start_consumer())

    async def stop(self) -> None:
        if not self._started:
            return

        self._producer_task.cancel()
        self._consumer_task.cancel()
        self._started = False

        return

    async def _producer(self):
        # read from the input track and stick it on a queue
        try:
            while self._started:
                frame : Frame = await self.input_track.recv()
                frame.dts = int(datetime.datetime.now().timestamp())
                await self.queue.put_latest(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("%s: Producer failed with exception: %s", self.name, e, exc_info=True)
            raise

    async def _start_consumer(
        self,
    ) -> None:
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
            while self._is_started:
                # Wait for at least one frame
                frame = await self.queue.get()
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
