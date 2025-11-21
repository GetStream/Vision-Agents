import asyncio
import logging
from typing import Optional

import av
from aiortc import MediaStreamTrack

logger = logging.getLogger(__name__)


class Sam3VideoTrack(MediaStreamTrack):
    """Video track that outputs SAM3-annotated frames."""
    
    kind = "video"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._ended = False
        logger.info("📹 Sam3VideoTrack initialized")

    async def add_frame(self, frame: av.VideoFrame):
        """Add a processed frame to the output queue."""
        if self._ended:
            return

        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                logger.warning("⚠️ Sam3VideoTrack queue full, dropping frame")

    async def recv(self) -> av.VideoFrame:
        """Receive the next frame from the queue."""
        if self._ended:
            raise MediaStreamError("Track ended")

        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            return frame
        except asyncio.TimeoutError:
            logger.warning("⚠️ Sam3VideoTrack recv timeout")
            raise MediaStreamError("Timeout waiting for frame")

    def stop(self):
        """Stop the video track."""
        super().stop()
        self._ended = True
        logger.info("🛑 Sam3VideoTrack stopped")


class MediaStreamError(Exception):
    """Exception raised for media stream errors."""
    pass
