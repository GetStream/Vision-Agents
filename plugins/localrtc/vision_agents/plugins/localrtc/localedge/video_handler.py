"""Video processing logic for LocalEdge.

This module handles video input track creation and management for local
video device access.
"""

import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class VideoHandler:
    """Handles video input tracks for LocalEdge.

    This class manages video device access and track creation, supporting
    both default device access and custom GStreamer pipelines.

    Attributes:
        video_device: Video input device identifier
        custom_pipeline: Optional GStreamer pipeline configuration
    """

    def __init__(
        self,
        video_device: Union[int, str],
        custom_pipeline: Optional[Dict[str, Any]],
    ) -> None:
        """Initialize the video handler.

        Args:
            video_device: Video input device identifier
            custom_pipeline: Optional GStreamer pipeline configuration
        """
        self.video_device = video_device
        self.custom_pipeline = custom_pipeline

        # Track reference
        self._video_input_track: Optional[Any] = None

    def create_video_input_track(self) -> Any:
        """Create a video input track for camera capture.

        Returns:
            VideoInputTrack or GStreamerVideoInputTrack instance
        """
        from ..tracks import GStreamerVideoInputTrack, VideoInputTrack

        if self._video_input_track is None and self.video_device is not None:
            logger.info(
                f"[LOCALRTC] Creating VideoInputTrack: device={self.video_device}"
            )

            if self.custom_pipeline and "video_source" in self.custom_pipeline:
                # Use GStreamer pipeline
                self._video_input_track = GStreamerVideoInputTrack(
                    pipeline=self.custom_pipeline["video_source"]
                )
            else:
                # Use default device access
                self._video_input_track = VideoInputTrack(device=self.video_device)

            logger.info("[LOCALRTC] Video input track created")

        return self._video_input_track

    def get_video_input_track(self) -> Optional[Any]:
        """Get the current video input track.

        Returns:
            Current video input track or None if not created
        """
        return self._video_input_track

    def stop_all_tracks(self) -> None:
        """Stop and cleanup all video tracks."""
        if self._video_input_track is not None:
            self._video_input_track.stop()
            self._video_input_track = None
            logger.info("[LOCALRTC] Video input track stopped")
