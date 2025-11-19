import asyncio
import logging
from typing import Optional

import av
from PIL import Image
from aiortc import MediaStreamTrack, VideoStreamTrack

from vision_agents.core.utils.video_queue import VideoLatestNQueue
from vision_agents.core.utils.video_utils import resize_frame

logger = logging.getLogger(__name__)


class DecartVideoTrack(VideoStreamTrack):
    """Video track that forwards Decart restyled video frames.

    Receives video frames from Decart's Realtime API and provides
    them through the standard VideoStreamTrack interface for publishing
    to the call.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        """Initialize the Decart video track.

        Args:
            width: Video frame width.
            height: Video frame height.
        """
        super().__init__()

        self.width = width
        self.height = height

        self.frame_queue: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(maxlen=2)
        placeholder = Image.new("RGB", (self.width, self.height), color=(30, 30, 40))
        self.placeholder_frame = av.VideoFrame.from_image(placeholder)
        self.last_frame: av.VideoFrame = self.placeholder_frame

        self._stopped = False
        self._source_track: Optional[MediaStreamTrack] = None

        logger.debug(f"DecartVideoTrack initialized ({width}x{height})")

    async def add_frame(self, frame: av.VideoFrame | av.AudioFrame | av.Packet) -> None:
        if self._stopped:
            return
        if not isinstance(frame, av.VideoFrame):
            return
        if frame.width != self.width or frame.height != self.height:
            frame = await asyncio.to_thread(resize_frame, self, frame)
        self.frame_queue.put_latest_nowait(frame)

    async def recv(self) -> av.VideoFrame:
        if self._stopped:
            raise ValueError("Track stopped")

        try:
            frame = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=0.033,
            )
            if frame:
                self.last_frame = frame
        except asyncio.TimeoutError:
            pass

        pts, time_base = await self.next_timestamp()

        output_frame = self.last_frame
        output_frame.pts = pts
        output_frame.time_base = time_base

        return output_frame

    @property
    def is_stopped(self) -> bool:
        """Check if the video track is stopped."""
        return self._stopped

    def stop(self) -> None:
        self._stopped = True
        super().stop()
