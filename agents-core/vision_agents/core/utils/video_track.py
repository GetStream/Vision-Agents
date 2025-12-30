import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, cast

import av
import av.filter
import av.frame
from aiortc import VideoStreamTrack
from av import VideoFrame
from PIL import Image
from vision_agents.core.utils.video_queue import VideoLatestNQueue
from vision_agents.core.utils.video_utils import ensure_even_dimensions

logger = logging.getLogger(__name__)


class VideoTrackClosedError(Exception): ...


class QueuedVideoTrack(VideoStreamTrack):
    """
    QueuedVideoTrack is an implementation of VideoStreamTrack that allows you to write video frames to it.
    It also gives you control over the width and height of the video frames.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 1,
        max_queue_size: int = 10,
    ):
        super().__init__()
        self.frame_queue: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(
            maxlen=max_queue_size
        )

        # Set video quality parameters
        self.width = width
        self.height = height
        self.fps = fps
        empty_image = Image.new("RGB", (self.width, self.height), color="blue")
        self.empty_frame = av.VideoFrame.from_image(empty_image)
        self.last_frame: av.VideoFrame = self.empty_frame
        self._stopped = False

    async def add_frame(self, frame: av.VideoFrame):
        if self._stopped:
            return

        # Ensure even dimensions for H.264 compatibility (screen shares often have odd dimensions)
        frame = ensure_even_dimensions(frame)

        self.frame_queue.put_latest_nowait(frame)

    async def recv(self) -> av.frame.Frame:
        """Receive the next video frame."""
        if self._stopped:
            raise VideoTrackClosedError("Track stopped")

        try:
            # Try to get a frame from queue with fps interval
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1 / self.fps)
            if frame:
                self.last_frame = frame
                logger.debug(f"📥 Got new frame from queue: {frame}")
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ Error getting frame from queue: {e}")

        # Get timestamp for the frame

        pts, time_base = await self.next_timestamp()

        # Create av.VideoFrame from PIL Image
        av_frame = self.last_frame

        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame

    def stop(self):
        self._stopped = True
        super(QueuedVideoTrack, self).stop()

    @property
    def stopped(self) -> bool:
        return self._stopped


class VideoFileTrack(VideoStreamTrack):
    """
    A video track reading from a local MP4 file,
    filtered to a constant FPS using FFmpeg (30 FPS by default).

    Use it for testing and debugging.
    """

    def __init__(self, path: str | Path, fps: int = 30):
        super().__init__()
        self.fps = fps
        self.path = Path(path)

        self._stopped = False
        self._frame_interval = 1.0 / self.fps
        self._container = av.open(path)

        if not self._container.streams.video:
            raise ValueError(f"No video streams found in file: {path}")

        self._stream = self._container.streams.video[0]
        if self._stream.time_base is None:
            raise ValueError("Cannot determine time_base for the video stream")

        self._time_base = self._stream.time_base

        # Decoder iterator to read the frames
        self._decoder = self._container.decode(self._stream)
        self._executor = ThreadPoolExecutor(1)
        self._set_filter_graph()

    def _set_filter_graph(self):
        # Safe extraction of sample_aspect_ratio
        sar = self._stream.sample_aspect_ratio
        if sar is None:
            sar_num, sar_den = 1, 1
        else:
            sar_num, sar_den = sar.numerator, sar.denominator

        # Build ffmpeg filter graph to resample video to fixed fps
        # Keep the reference to the graph to avoid GC
        self._graph = av.filter.Graph()
        # Buffer source with all required parameters

        self._src = self._graph.add(
            "buffer",
            f"video_size={self._stream.width}x{self._stream.height}:"
            f"pix_fmt={self._stream.pix_fmt}:"
            f"time_base={self._time_base.numerator}/{self._time_base.denominator}:"
            f"pixel_aspect={sar_num}/{sar_den}",
        )

        # Add an FPS filter
        fps_filter = self._graph.add("fps", f"fps={self.fps}")

        # Add a buffer sink
        self._sink = self._graph.add("buffersink")

        # Connect graph: buffer -> fps filter -> sink
        self._src.link_to(fps_filter)
        fps_filter.link_to(self._sink)
        self._graph.configure()

    def _next_frame(self) -> av.VideoFrame:
        filtered_frame: Optional[av.VideoFrame] = None
        while filtered_frame is None:
            # Get the next decoded frame
            try:
                frame = next(self._decoder)
            except StopIteration:
                # Loop the video when it ends
                self._container.seek(0)
                self._decoder = self._container.decode(self._stream)
                # Reset the filter graph too
                self._set_filter_graph()
                frame = next(self._decoder)

            # Ensure frame has a time_base (required by buffer source)
            frame.time_base = self._time_base

            # Push decoded frame into the filter graph
            self._src.push(frame)

            # Pull filtered frame from buffersink
            try:
                filtered_frame = cast(av.VideoFrame, self._sink.pull())
            except (av.ExitError, av.BlockingIOError):
                # Filter graph is not ready to output yet
                time.sleep(0.001)
                continue
            except Exception:
                logger.exception(
                    f'Failed to read a video frame from file "{self.path}"'
                )
                time.sleep(0.001)
                continue

        # Convert the filtered video frame to RGB for aiortc
        new_frame = filtered_frame.to_rgb()

        return new_frame

    async def recv(self) -> VideoFrame:
        """
        Async method to produce the next filtered video frame.
        Loops automatically at the end of the file.
        """
        if self._stopped:
            raise VideoTrackClosedError("Track stopped")
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(self._executor, self._next_frame)

        # Sleep between frames to simulate real-time playback
        await asyncio.sleep(self._frame_interval)
        return frame

    def stop(self) -> None:
        self._stopped = True
        self._executor.shutdown(wait=False)
        self._container.close()
        super(VideoFileTrack, self).stop()

    def __repr__(self):
        return f'<{self.__class__.__name__} path="{self.path}" fps={self.fps}>'
