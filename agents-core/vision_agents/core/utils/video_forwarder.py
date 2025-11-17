import asyncio
import datetime
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Any, Literal

import av
from aiortc import VideoStreamTrack
from av.frame import Frame
from PIL import Image

from vision_agents.core.utils.video_queue import VideoLatestNQueue

logger = logging.getLogger(__name__)

ResamplingMethod = Literal["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"]


@dataclass
class FrameHandler:
    """Handler configuration for processing video frames."""
    callback: Callable[[av.VideoFrame], Any]
    fps: Optional[float]
    name: str
    last_ts: float = 0.0
    width: Optional[int] = None  # None = no resizing
    height: Optional[int] = None  # None = no resizing
    resampling: ResamplingMethod = "LANCZOS"

class VideoForwarder:
    """
    VideoForwarder handles forwarding a video track to 1 or multiple targets.
    
    Each handler can specify its own frame rate, dimensions, and resizing quality.
    This allows sending different sized frames to different targets efficiently.
    
    By default, no resizing is performed - frames are passed through as-is.
    Specify width and/or height to enable resizing for a handler.

    Example:

        forwarder = VideoForwarder(input_track=track, fps=5)
        
        # Add handler without resizing (passes original frames)
        forwarder.add_frame_handler(process_frame, fps=1)
        
        # Add handler with custom resizing for OpenAI (720p, high quality)
        forwarder.add_frame_handler(
            send_to_openai, 
            fps=1, 
            width=1280, 
            height=720,
            resampling="LANCZOS"
        )
        
        # Add handler with different size for thumbnails (low res, fast)
        forwarder.add_frame_handler(
            save_thumbnail, 
            fps=0.5, 
            width=320, 
            height=180,
            resampling="BILINEAR"
        )
        
        # Starts automatically when attaching handlers
        # Stop when done:
        await forwarder.stop()

    """
    def __init__(self, input_track: VideoStreamTrack, *, max_buffer: int = 10, fps: Optional[float] = 30, name: str = "video-forwarder"):
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
            width: Optional[int] = None,
            height: Optional[int] = None,
            resampling: ResamplingMethod = "LANCZOS",
    ) -> None:
        """
        Register a callback to be called for each frame.

        Args:
            on_frame: Callback function (sync or async) to receive frames
            fps: Frame rate for this handler (overrides default). None = unlimited.
            name: Optional name for this handler (for logging)
            width: Optional target width for resizing. If None, no resizing on width.
            height: Optional target height for resizing. If None, no resizing on height.
            resampling: Resampling method for resize: "NEAREST", "BILINEAR", "BICUBIC", or "LANCZOS" (only used if resizing)
        """
        handler_name = name or f"handler-{len(self._frame_handlers)}"
        handler_fps = fps if fps is not None else self.fps
        if fps is not None and self.fps is not None and fps > self.fps:
            raise ValueError(f"fps on handler {fps} cannot be greater than fps on forwarder {self.fps}")

        handler = FrameHandler(
            callback=on_frame,
            fps=handler_fps,
            name=handler_name,
            width=width,
            height=height,
            resampling=resampling,
        )
        self._frame_handlers.append(handler)
        self.start()

    async def remove_frame_handler(self, on_frame: Callable[[av.VideoFrame], Any]) -> bool:
        """
        Remove a previously registered callback.

        Args:
            on_frame: The callback to remove

        Returns:
            True if the handler was found and removed, False otherwise
        """
        original_len = len(self._frame_handlers)
        self._frame_handlers = [h for h in self._frame_handlers if h.callback != on_frame]
        removed = len(self._frame_handlers) < original_len

        if len(self._frame_handlers) == 0:
            await self.stop()
        return removed

    def start(self) -> None:
        """Start the producer and consumer tasks if not already started."""
        if self._started:
            return
        self._started = True
        self._producer_task = asyncio.create_task(self._producer())
        self._consumer_task = asyncio.create_task(self._start_consumer())

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
                frame : Frame = await self.input_track.recv()
                frame.dts = int(datetime.datetime.now().timestamp())
                await self.queue.put_latest(frame)
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
                    min_interval = (1.0 / handler.fps) if (handler.fps and handler.fps > 0) else 0.0
                    
                    # Check if enough time has passed for this handler
                    if min_interval == 0.0 or (now - handler.last_ts) >= min_interval:
                        handler.last_ts = now
                        
                        # Resize frame if dimensions are specified
                        processed_frame = frame
                        if handler.width is not None or handler.height is not None:
                            target_width = handler.width or frame.width
                            target_height = handler.height or frame.height
                            
                            if frame.width != target_width or frame.height != target_height:
                                processed_frame = await self._resize_frame(
                                    frame, target_width, target_height, 
                                    handler.resampling
                                )
                        
                        # Call handler (sync or async)
                        if asyncio.iscoroutinefunction(handler.callback):
                            await handler.callback(processed_frame)
                        else:
                            handler.callback(processed_frame)
        except asyncio.CancelledError:
            raise
    
    async def _resize_frame(
        self, 
        frame: av.VideoFrame, 
        width: int, 
        height: int, 
        resampling: ResamplingMethod
    ) -> av.VideoFrame:
        """Resize a frame using the specified resampling method."""
        def _do_resize():
            # Map string to PIL constant
            resampling_map = {
                "NEAREST": Image.Resampling.NEAREST,
                "BILINEAR": Image.Resampling.BILINEAR,
                "BICUBIC": Image.Resampling.BICUBIC,
                "LANCZOS": Image.Resampling.LANCZOS,
            }
            
            pil_image = frame.to_image()
            resized = pil_image.resize((width, height), resampling_map[resampling])
            
            return av.VideoFrame.from_image(resized)
        
        return await asyncio.to_thread(_do_resize)

