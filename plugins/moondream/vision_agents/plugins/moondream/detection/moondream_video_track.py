import asyncio
import logging

import av
from PIL import Image
from aiortc import VideoStreamTrack

from vision_agents.core.utils.video_queue import VideoLatestNQueue

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class MoondreamVideoTrack(VideoStreamTrack):
    """Video track for publishing Moondream-processed frames.

    Uses a LatestNQueue to buffer processed frames and publishes them
    at the configured frame rate.

    Args:
        width: Frame width in pixels (default: 640)
        height: Frame height in pixels (default: 480)
    """

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        super().__init__()
        logger.info("MoondreamVideoTrack: initializing")
        self.frame_queue: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(
            maxlen=10
        )

        # Set video quality parameters
        self.width = width
        self.height = height
        empty_image = Image.new("RGB", (self.width, self.height), color="blue")
        self.empty_frame = av.VideoFrame.from_image(empty_image)
        self.last_frame: av.VideoFrame = self.empty_frame
        self._stopped = False

    async def add_frame(self, frame: av.VideoFrame):
        if self._stopped:
            return

        self.frame_queue.put_latest_nowait(frame)

    async def recv(self) -> av.frame.Frame:
        """
        Receive the next video frame for publishing.

        Returns:
            Video frame with proper PTS and time_base
        """
        if self._stopped:
            raise Exception("Track stopped")

        try:
            # Try to get a frame from queue with short timeout
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.02)
            if frame:
                self.last_frame = frame
                logger.debug("📥 Got new frame from queue")
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ Error getting frame from queue: {e}")

        # Get timestamp for the frame
        pts, time_base = await self.next_timestamp()

        # Create av.VideoFrame from last frame
        av_frame = self.last_frame
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame

    def stop(self):
        """Stop the video track."""
        self._stopped = True
