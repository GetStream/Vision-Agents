import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Dict, List

import aiortc
import av
import cv2
import numpy as np
from PIL import Image

from vision_agents.core.processors.base_processor import (
    VideoProcessorMixin,
    VideoPublisherMixin,
    AudioVideoProcessor,
)
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)


class RoboflowVideoTrack(QueuedVideoTrack):
    """Video track for Roboflow processed frames."""
    pass


class RoboflowDetectionProcessor(
    AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin
):
    """
    Real-time object detection using Roboflow Universe public models.
    
    Loads models from Roboflow Universe. If version is not specified, uses latest.
    
    Model ID Format:
        workspace/project or workspace/project/version
    
    Find models at: https://universe.roboflow.com
    From URL like: universe.roboflow.com/workspace/project
    Use model_id: "workspace/project"
    
    Args:
        model_id: Universe model in format "workspace/project" or "workspace/project/version"
        version: Model version (optional, defaults to latest)
        api_key: Roboflow API key (required, get from app.roboflow.com)
        conf_threshold: Confidence threshold for detections (0-100)
        fps: Frame processing rate (default: 5 to be API-friendly)
        interval: Processing interval in seconds (0 = process every frame)
        max_workers: Number of worker threads for async processing
    
    Examples:
        # Latest version
        processor = RoboflowDetectionProcessor(
            model_id="roboflow-100/aerial-spheres"
        )
        
        # Specific version in model_id
        processor = RoboflowDetectionProcessor(
            model_id="roboflow-100/aerial-spheres/2"
        )
        
        # Specific version as parameter
        processor = RoboflowDetectionProcessor(
            model_id="roboflow-100/aerial-spheres",
            version=2
        )
    """

    def __init__(
        self,
        model_id: str,
        version: Optional[int] = None,
        api_key: Optional[str] = None,
        conf_threshold: int = 40,
        fps: int = 5,
        interval: int = 0,
        max_workers: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)

        if not model_id:
            raise ValueError("model_id is required")

        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.model_id = model_id
        self.version = version
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.max_workers = max_workers
        self._shutdown = False
        self._video_forwarder: Optional[VideoForwarder] = None

        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="roboflow_processor"
        )

        # Video track for publishing
        self._video_track: RoboflowVideoTrack = RoboflowVideoTrack()

        # Initialize Roboflow
        self._load_model()

        logger.info("🔍 Roboflow Processor initialized")
        logger.info(f"📦 Using model: {model_id}" + (f" v{version}" if version else ""))

    def _load_model(self):
        """
        Load a public model from Roboflow Universe.
        
        Format: workspace/project or workspace/project/version
        """
        try:
            from roboflow import Roboflow

            if not self.api_key:
                raise ValueError(
                    "ROBOFLOW_API_KEY required. Get it from https://app.roboflow.com → Settings → API"
                )

            parts = self.model_id.split("/")
            
            if len(parts) == 2:
                # Format: workspace/project
                workspace_id, project_id = parts
                version_num = self.version
                
            elif len(parts) == 3:
                # Format: workspace/project/version
                workspace_id, project_id, version_str = parts
                version_num = int(version_str)
                if self.version and self.version != version_num:
                    logger.warning(f"⚠️ Using version from model_id: {version_num}")
                    
            else:
                raise ValueError(
                    f"Invalid model_id: '{self.model_id}'\n"
                    f"Expected format: 'workspace/project' or 'workspace/project/version'\n"
                    f"Example: 'roboflow-100/aerial-spheres' or 'roboflow-100/aerial-spheres/2'\n"
                    f"Find models at: https://universe.roboflow.com"
                )
            
            # Load model
            rf = Roboflow(api_key=self.api_key)
            workspace = rf.workspace(workspace_id)
            project = workspace.project(project_id)
            
            # Get version
            if version_num is None:
                versions = project.versions()
                if not versions:
                    raise ValueError(f"No versions found for {self.model_id}")
                version_num = versions[-1].version
                logger.info(f"📌 Using latest version: {version_num}")
            
            self.version = version_num
            self.model = project.version(version_num).model
            
            logger.info(f"✅ Loaded model: {workspace_id}/{project_id} v{version_num}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model '{self.model_id}': {e}")
            logger.error("Check: https://universe.roboflow.com to find the correct model_id")
            raise

    async def process_video(
        self,
        incoming_track: aiortc.MediaStreamTrack,
        participant: Any,
        shared_forwarder=None,
    ):
        """Process incoming video track with Roboflow detection."""
        logger.info("✅ Roboflow process_video starting")

        if shared_forwarder is not None:
            # Use shared VideoForwarder
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Roboflow subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            await self._video_forwarder.start_event_consumer(
                self._process_and_add_frame,
                fps=float(self.fps),
                consumer_name="roboflow",
            )
        else:
            # Create own VideoForwarder
            self._video_forwarder = VideoForwarder(
                incoming_track,
                max_buffer=30,
                fps=self.fps,
                name="roboflow_forwarder",
            )

            await self._video_forwarder.start()
            await self._video_forwarder.start_event_consumer(
                self._process_and_add_frame
            )

        logger.info("✅ Roboflow video processing pipeline started")

    def publish_video_track(self):
        """Return the video track for publishing processed frames."""
        logger.info("📹 publish_video_track called")
        return self._video_track

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        """Process frame, run detection, annotate, and publish."""
        try:
            frame_array = frame.to_ndarray(format="rgb24")

            # Run inference
            results = await self._run_inference(frame_array)

            # Annotate frame with detections
            if results.get("predictions"):
                frame_array = self._annotate_frame(
                    frame_array, results["predictions"]
                )

            # Convert back to av.VideoFrame and publish
            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)

    async def _run_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """Run Roboflow inference on frame."""
        try:
            # Convert to PIL Image
            image = Image.fromarray(frame_array)

            # Run inference in thread pool (Roboflow SDK is synchronous)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._run_detection_sync, image
            )

            return result
        except Exception as e:
            logger.exception(f"❌ Roboflow inference failed: {e}")
            return {"predictions": []}

    def _run_detection_sync(self, image: Image.Image) -> Dict[str, Any]:
        """Synchronous detection call."""
        if self._shutdown:
            return {"predictions": []}

        temp_file = None
        try:
            # Roboflow needs a file path, so save to temp file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name, format="JPEG")
                temp_path = temp_file.name
            
            # Run inference
            result = self.model.predict(temp_path, confidence=self.conf_threshold)

            # Parse result - Roboflow returns a response object
            predictions = result.json().get("predictions", [])

            logger.debug(f"🔍 Roboflow detected {len(predictions)} objects")
            return {"predictions": predictions}
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return {"predictions": []}
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    def _annotate_frame(
        self, frame_array: np.ndarray, predictions: List[Dict]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        height, width = frame_array.shape[:2]

        for pred in predictions:
            # Roboflow uses center x, y, width, height format
            cx = pred.get("x", 0)
            cy = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)

            # Convert to corner coordinates
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Clamp to frame boundaries
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Draw bounding box
            cv2.rectangle(frame_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            class_name = pred.get("class", "object")
            confidence = pred.get("confidence", 0.0)
            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                frame_array,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame_array

    def close(self):
        """Clean up resources."""
        self._shutdown = True
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
        if self._video_track:
            self._video_track.stop()
        logger.info("🛑 Roboflow Processor closed")

