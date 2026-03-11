"""GridOverlayProcessor — draws a labeled grid on screen share frames."""

import logging
from typing import Optional

import aiortc
import av
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

from ._grid import Grid

logger = logging.getLogger(__name__)


class GridOverlayProcessor(VideoProcessorPublisher):
    """Draws a configurable grid on incoming video frames.

    The annotated frames are forwarded to the LLM so it can reference
    grid cells (e.g. "C2") instead of guessing raw pixel coordinates.

    Args:
        grid: Shared Grid instance. If provided, cols/rows are ignored.
        cols: Number of grid columns (1-26). Default 15.
        rows: Number of grid rows (1-99). Default 15.
        fps: Frame rate for processing. Default 2.

    Usage::

        from vision_agents.plugins.computer_use import Grid, GridOverlayProcessor

        grid = Grid(cols=20, rows=20)
        agent = Agent(
            ...,
            processors=[GridOverlayProcessor(grid=grid)],
        )
    """

    name = "grid_overlay"

    def __init__(
        self,
        grid: Grid | None = None,
        cols: int = 15,
        rows: int = 15,
        fps: float = 2,
    ):
        self._grid = grid if grid is not None else Grid(cols=cols, rows=rows)
        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None
        self._video_track = QueuedVideoTrack()
        self._shutdown = False

    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        return self._video_track

    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._on_frame)

        logger.info(
            "Starting grid overlay processor at %.1f FPS (grid %s)",
            self._fps,
            self._grid.label,
        )
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                track,
                max_buffer=int(self._fps),
                fps=self._fps,
                name="grid_overlay_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._on_frame, fps=self._fps, name="grid_overlay"
        )

    async def _on_frame(self, frame: av.VideoFrame) -> None:
        if self._shutdown:
            return
        try:
            annotated = self._grid.draw_overlay(frame)
        except Exception:
            logger.exception("draw_overlay failed, forwarding original frame")
            annotated = frame
        await self._video_track.add_frame(annotated)

    async def stop_processing(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._on_frame)
            self._video_forwarder = None
            logger.info("Stopped grid overlay processor")

    async def close(self) -> None:
        self._shutdown = True
        await self.stop_processing()
