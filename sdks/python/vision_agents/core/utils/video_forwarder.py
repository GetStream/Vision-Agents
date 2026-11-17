import asyncio
import datetime
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import av
from aiortc import MediaStreamError, VideoStreamTrack
from av.frame import Frame
from vision_agents.core.utils.video_queue import VideoLatestNQueue

logger = logging.getLogger(__name__)


@dataclass
class FrameHandler:
    """Handler configuration for processing video frames."""

    callback: Callable[[av.VideoFrame], Any]
    fps: Optional[float]
    name: str
    last_ts: float = 0.0


class VideoForwarder:
    """
    VideoForwarder handles forwarding a video track to 1 or multiple targets

    Example:

        forwarder = VideoForwarder(input_track=track, fps=5)
        forwarder.add_frame_handler( lamba x: print("received frame"), fps =1 )
        forwarder.stop()

        # start's automatically when attaching handlers

    """

    def __init__(
        self,
        input_track: VideoStreamTrack,
        *,
        max_buffer: int = 10,
        fps: Optional[float] = 30,
        name: str = "video-forwarder",
    ):
        self.name = name
        self.input_track = input_track
        self.queue: VideoLatestNQueue[Frame] = VideoLatestNQueue(maxlen=max_buffer)
        self.fps = fps  # None = unlimited, else forward at ~fps

        self._producer_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._frame_handlers: list[FrameHandler] = []
        self._started = False

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
        handler_fps = fps if fps is not None else self.fps
        if fps is not None and self.fps is not None and fps > self.fps:
            raise ValueError(
                f"fps on handler {fps} cannot be greater than fps on forwarder {self.fps}"
            )

        handler = FrameHandler(
            callback=on_frame,
            fps=handler_fps,
            name=handler_name,
        )
        self._frame_handlers.append(handler)
        self.start()

    async def remove_frame_handler(
        self, on_frame: Callable[[av.VideoFrame], Any]
    ) -> bool:
        """
        Remove a previously registered callback.

        Args:
            on_frame: The callback to remove

        Returns:
            True if the handler was found and removed, False otherwise
        """
        original_len = len(self._frame_handlers)
        self._frame_handlers = [
            h for h in self._frame_handlers if h.callback != on_frame
        ]
        removed = len(self._frame_handlers) < original_len

        if len(self._frame_handlers) == 0:
            await self.stop()
        return removed

    @property
    def frame_handlers(self) -> list[FrameHandler]:
        return self._frame_handlers

    def start(self) -> None:
        """Start the producer and consumer tasks if not already started."""
        if self._started:
            return
        self._started = True
        self._producer_task = asyncio.create_task(self._producer())
        self._consumer_task = asyncio.create_task(self._start_consumer())

    @property
    def started(self) -> bool:
        return self._started

    async def stop(self) -> None:
        if not self._started:
            return

        if self._producer_task is not None:
            self._producer_task.cancel()
        if self._consumer_task is not None:
            self._consumer_task.cancel()
        self._started = False

        return

    async def _producer(self):
        # read from the input track and stick it on a queue
        try:
            while self._started:
                frame: Frame = await self.input_track.recv()
                frame.dts = int(datetime.datetime.now().timestamp())
                await self.queue.put_latest(frame)
        except MediaStreamError:
            # Raise errors only if the media track is still live.
            if self.input_track.readyState == "live":
                raise
        except asyncio.CancelledError:
            raise

    async def _start_consumer(self) -> None:
        """Consumer loop that forwards frames to all registered handlers."""
        loop = asyncio.get_running_loop()

        try:
            while self._started:
                frame = await self.queue.get()
                now = loop.time()

                # Call each handler if enough time has passed per its fps setting
                for handler in self._frame_handlers:
                    min_interval = (
                        (1.0 / handler.fps)
                        if (handler.fps and handler.fps > 0)
                        else 0.0
                    )

                    # Check if enough time has passed for this handler
                    if min_interval == 0.0 or (now - handler.last_ts) >= min_interval:
                        handler.last_ts = now

                        # Call handler (sync or async)
                        try:
                            if asyncio.iscoroutinefunction(handler.callback):
                                await handler.callback(frame)
                            else:
                                handler.callback(frame)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                f"Frame handler {handler.name} failed with an exception"
                            )
        except asyncio.CancelledError:
            raise
