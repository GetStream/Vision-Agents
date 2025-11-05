import logging
import os
from typing import List, Optional, Union

import aiortc

from vision_agents.core import (
    llm
)

from vision_agents.core.utils.video_forwarder import VideoForwarder
import moondream as md


logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class CloudVLM(llm.VideoLLM):
    """
    Using the CloudVLM, you can send frames to the hosted Moondream model to perform either captioning or Visual queries.
    The instructions are taken from the STT service and sent to the model along with the frame. Once the model has an output, the results are then vocalised with the supplied TTS service.

    You can specify whether to use the caption endpoint or query (VQA).
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            conf_threshold: float = 0.3,
            mode: str = "caption", # Possible values - local, vqa
            detect_objects: Union[str, List[str]] = "person",
            max_workers: int = 10,
    ):
        super().__init__()

        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY")
        self.conf_threshold = conf_threshold
        self.max_workers = max_workers
        self._shutdown = False

        # Initialize model
        self._load_model()

    async def watch_video_track(self, track: aiortc.mediastreams.MediaStreamTrack,
                                shared_forwarder: Optional[VideoForwarder] = None) -> None:
        pass


    def _load_model(self):
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("api_key is required for Moondream Cloud API")

            # Initialize cloud model
            self.model = md.vl(api_key=self.api_key)
            logger.info("✅ Moondream SDK initialized")

        except Exception as e:
            logger.exception(f"❌ Failed to load Moondream model: {e}")
            raise


    def close(self):
        """Clean up resources."""
        self._shutdown = True
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
        logger.info("🛑 Moondream Processor closed")

