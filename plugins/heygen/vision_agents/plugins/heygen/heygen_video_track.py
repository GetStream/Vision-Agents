import asyncio
import logging
from typing import Optional

import av
from aiortc import MediaStreamTrack, VideoStreamTrack
from PIL import Image

from vision_agents.core.utils.video_queue import VideoLatestNQueue

logger = logging.getLogger(__name__)


class HeyGenVideoTrack(VideoStreamTrack):
    """Video track that forwards HeyGen avatar video frames.

    Receives video frames from HeyGen's WebRTC connection and provides
    them through the standard VideoStreamTrack interface for publishing
    to the call.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        """Initialize the HeyGen video track.

        Args:
            width: Video frame width.
            height: Video frame height.
        """
        super().__init__()

        self.width = width
        self.height = height

        # Queue for incoming frames from HeyGen - keep minimal for low latency
        self.frame_queue: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(maxlen=2)

        # Create placeholder frame for when no frames are available
        placeholder = Image.new("RGB", (self.width, self.height), color=(30, 30, 40))
        self.placeholder_frame = av.VideoFrame.from_image(placeholder)
        self.last_frame: av.VideoFrame = self.placeholder_frame

        self._stopped = False
        self._receiving_task: Optional[asyncio.Task] = None
        self._source_track: Optional[MediaStreamTrack] = None

        logger.info(f"HeyGenVideoTrack initialized ({width}x{height})")

    async def start_receiving(self, source_track: MediaStreamTrack) -> None:
        """Start receiving frames from HeyGen's video track.

        Args:
            source_track: The incoming video track from HeyGen's WebRTC connection.
        """
        if self._receiving_task:
            logger.info("Restarting HeyGen video receiver with new source track")
            self._receiving_task.cancel()
            try:
                await self._receiving_task
            except asyncio.CancelledError:
                pass
            self._receiving_task = None
            self._source_track = None

        self._source_track = source_track
        self._receiving_task = asyncio.create_task(self._receive_frames())
        logger.info("Started receiving frames from HeyGen")

    async def _receive_frames(self) -> None:
        """Continuously receive frames from HeyGen and add to queue."""
        if not self._source_track:
            logger.error("No source track set")
            return

        try:
            while not self._stopped:
                try:
                    # Receive frame from HeyGen
                    frame = await self._source_track.recv()

                    # Type check: ensure we have a VideoFrame
                    if frame and isinstance(frame, av.VideoFrame):
                        # Resize if needed
                        if frame.width != self.width or frame.height != self.height:
                            frame = self._resize_frame(frame)

                        # Add to queue (will replace oldest if full)
                        self.frame_queue.put_latest_nowait(frame)

                        logger.debug(
                            f"Received frame from HeyGen: {frame.width}x{frame.height}"
                        )

                except Exception as e:
                    if not self._stopped:
                        logger.warning(f"Error receiving frame from HeyGen: {e}")
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("Frame receiving task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in frame receiving: {e}")

    def _resize_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Resize a video frame to match the track dimensions while maintaining aspect ratio.

        Args:
            frame: Input video frame.

        Returns:
            Resized video frame with letterboxing if needed.
        """
        try:
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

        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame

    async def recv(self) -> av.VideoFrame:
        """Receive the next video frame.

        This is called by the WebRTC stack to get frames for transmission.

        Returns:
            Video frame to transmit.
        """
        if self._stopped:
            raise Exception("Track stopped")

        try:
            # Try to get a new frame from queue with short timeout
            frame = await asyncio.wait_for(
                self.frame_queue.get(),
                timeout=0.033,  # ~30 FPS
            )
            if frame:
                self.last_frame = frame

        except asyncio.TimeoutError:
            # No new frame, use last frame
            pass

        except Exception as e:
            logger.warning(f"Error getting frame from queue: {e}")

        # Get timestamp for the frame
        pts, time_base = await self.next_timestamp()

        # Create a copy of the frame with updated timestamp
        output_frame = self.last_frame
        output_frame.pts = pts
        output_frame.time_base = time_base

        return output_frame

    def stop(self) -> None:
        """Stop the video track."""
        self._stopped = True

        if self._receiving_task:
            self._receiving_task.cancel()
            self._receiving_task = None

        super().stop()
        logger.info("HeyGenVideoTrack stopped")
