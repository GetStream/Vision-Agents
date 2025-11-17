import asyncio
import logging
from typing import Optional

import av
from PIL import Image
from aiortc import MediaStreamTrack, VideoStreamTrack

from vision_agents.core.utils.video_queue import VideoLatestNQueue

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

    async def add_frame(self, frame: av.VideoFrame) -> None:
        if self._stopped:
            return
        # if frame.width != self.width or frame.height != self.height:
        #     frame = await asyncio.to_thread(self._resize_frame, frame)
        self.frame_queue.put_latest_nowait(frame)

    # TODO: move this to a utils file
    def _resize_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        logger.debug(
            f"Resizing frame from {frame.width}x{frame.height} to {self.width}x{self.height}"
        )
        img = frame.to_image()

        # Calculate scaling to maintain aspect ratio
        src_width, src_height = img.size
        target_width, target_height = self.width, self.height

        # Calculate scale factor (fit within target dimensions)
        scale = min(target_width / src_width, target_height / src_height)
        new_width = int(src_width * scale)
        new_height = int(src_height * scale)

        # Resize with aspect ratio maintained
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create black background at target resolution
        result = Image.new("RGB", (target_width, target_height), (0, 0, 0))

        # Paste resized image centered
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(resized, (x_offset, y_offset))

        return av.VideoFrame.from_image(result)

    async def recv(self) -> av.VideoFrame:
        if self._stopped:
            raise Exception("Track stopped")

        try:
            frame = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=0.033,  # ~30 FPS
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

    def stop(self) -> None:
        self._stopped = True
        super().stop()
