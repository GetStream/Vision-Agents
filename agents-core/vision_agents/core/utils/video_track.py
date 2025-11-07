import asyncio

import av
from PIL import Image
from aiortc import VideoStreamTrack

from vision_agents.core.utils.video_queue import VideoLatestNQueue
import logging

logger = logging.getLogger(__name__)

class QueuedVideoTrack(VideoStreamTrack):
    """
    QueuedVideoTrack is an implementation of VideoStreamTrack that allows you write video frames to it
    It also gives you control over the width and height of the video frames
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 1):
        super().__init__()
        self.frame_queue: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(maxlen=10)

        # Set video quality parameters
        self.width = width
        self.height = height
        self.fps = fps
        empty_image = Image.new("RGB", (self.width, self.height), color="blue")
        self.empty_frame = av.VideoFrame.from_image(empty_image)
        self.last_frame: av.VideoFrame = self.empty_frame
        self._stopped = False

    async def add_frame(self, frame: av.VideoFrame):
        # Resize the image and stick it on the queue
        if self._stopped:
            return

        # TODO: where do we resize? do we need to resize?...
        # Ensure the image is the correct size
        # if image.size != (self.width, self.height):
        #    image = image.resize(
        #        (self.width, self.height), Image.Resampling.BILINEAR
        #    )

        self.frame_queue.put_latest_nowait(frame)

    async def recv(self) -> av.frame.Frame:
        """Receive the next video frame."""
        if self._stopped:
            raise Exception("Track stopped")

        try:
            # Try to get a frame from queue with fps interval
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1/self.fps)
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