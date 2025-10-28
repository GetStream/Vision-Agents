import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import aiortc
import av
import cv2
import numpy as np
from aiortc import VideoStreamTrack
from PIL import Image

from vision_agents.core.processors.base_processor import (
    VideoProcessorMixin,
    VideoPublisherMixin,
    AudioVideoProcessor,
)
from vision_agents.core.utils.queue import LatestNQueue
from vision_agents.core.utils.video_forwarder import VideoForwarder
import moondream as md


logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class MoondreamVideoTrack(VideoStreamTrack):
    """
    Video track for publishing Moondream-processed frames.
    
    Uses a LatestNQueue to buffer processed frames and publishes them
    at the configured frame rate.
    """
    
    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        super().__init__()
        logger.info("MoondreamVideoTrack: initializing")
        self.frame_queue: LatestNQueue[av.VideoFrame] = LatestNQueue(maxlen=10)
        
        # Set video quality parameters
        self.width = width
        self.height = height
        empty_image = Image.new("RGB", (self.width, self.height), color="blue")
        self.empty_frame = av.VideoFrame.from_image(empty_image)
        self.last_frame: av.VideoFrame = self.empty_frame
        self._stopped = False
    
    async def add_frame(self, frame: av.VideoFrame):
        """
        Add a processed frame to the queue for publishing.
        
        Args:
            frame: Processed video frame with annotations
        """
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
                logger.debug(f"📥 Got new frame from queue")
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


class MoondreamProcessor(AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin):
    """Performs real-time object detection on video streams using Moondream Cloud API.
    
    Args:
        api_key: API key for Moondream Cloud API. If not provided, will attempt to read
                from MOONDREAM_API_KEY environment variable.
        conf_threshold: Confidence threshold for detections
        detect_objects: Object(s) to detect. Moondream uses zero-shot detection,
                       so any object string works. Examples: "person", "car",
                       "basketball", ["person", "car", "dog"]. Default: "person"
        fps: Frame processing rate
        interval: Processing interval in seconds
        max_workers: Number of worker threads
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        conf_threshold: float = 0.3,
        detect_objects: Union[str, List[str]] = "person",
        fps: int = 30,
        interval: int = 0,
        max_workers: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)
        
        # Auto-load API key from environment if not provided
        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY")
        
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.max_workers = max_workers
        self._shutdown = False
        
        # Normalize detect_objects to list of strings
        if isinstance(detect_objects, str):
            self.detect_objects = [detect_objects]
        elif isinstance(detect_objects, list):
            if not all(isinstance(obj, str) for obj in detect_objects):
                raise ValueError("detect_objects must be str or list of strings")
            self.detect_objects = detect_objects
        else:
            raise ValueError("detect_objects must be str or list of strings")
        
        # Thread pool for CPU-intensive inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="moondream_processor"
        )
        
        # Video track for publishing (if used as video publisher)
        self._video_track: MoondreamVideoTrack = MoondreamVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None
        
        # Initialize model
        self._load_model()
        
        logger.info("🌙 Moondream Processor initialized")
        logger.info(f"🎯 Detection configured for objects: {self.detect_objects}")
    
    async def process_video(
        self,
        incoming_track: aiortc.mediastreams.MediaStreamTrack,
        participant: Any,
        shared_forwarder=None,
    ):
        """
        Process incoming video track.
        
        This method sets up the video processing pipeline:
        1. Uses shared VideoForwarder if provided, otherwise creates own
        2. Starts event consumer that calls _process_and_add_frame for each frame
        3. Frames are processed, annotated, and published via the video track
        
        Args:
            incoming_track: Incoming video stream from participant
            participant: Participant information
            shared_forwarder: Optional shared VideoForwarder to use
        """
        logger.info("✅ Moondream process_video starting")
        
        if shared_forwarder is not None:
            # Use the shared forwarder
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Moondream subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            await self._video_forwarder.start_event_consumer(
                self._process_and_add_frame,
                fps=float(self.fps),
                consumer_name="moondream"
            )
        else:
            # Create our own VideoForwarder
            self._video_forwarder = VideoForwarder(
                incoming_track,  # type: ignore[arg-type]
                max_buffer=30,  # 1 second at 30fps
                fps=self.fps,
                name="moondream_forwarder",
            )
            
            # Start the forwarder
            await self._video_forwarder.start()
            await self._video_forwarder.start_event_consumer(
                self._process_and_add_frame
            )
            
        logger.info("✅ Moondream video processing pipeline started")
    
    def publish_video_track(self):
        """
        Publish processed video track with annotations.
        
        Returns:
            VideoStreamTrack with Moondream annotations
        """
        logger.info("📹 publish_video_track called")
        return self._video_track
    
    def _load_model(self):
        """Load Moondream model using the official SDK."""
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("api_key is required for Moondream Cloud API")
            
            # Initialize cloud model
            self.model = md.vl(api_key=self.api_key)
            logger.info("✅ Moondream SDK initialized")
                
        except ImportError:
            raise ImportError(
                "moondream SDK is not installed. "
                "Please install it: pip install moondream"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load Moondream model: {e}")
            raise
    
    async def _run_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """
        Run inference using Moondream SDK.
        
        Args:
            frame_array: Input frame as numpy array (RGB format)
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Convert frame to PIL Image
            image = Image.fromarray(frame_array)
            
            # Call SDK for each object type
            # The SDK's detect() is synchronous, so wrap in executor
            loop = asyncio.get_event_loop()
            all_detections = await loop.run_in_executor(
                self.executor, self._run_detection_sync, image
            )
            
            return {"detections": all_detections}
        except Exception as e:
            logger.error(f"❌ Cloud inference failed: {e}")
            return {}
    
    def _run_detection_sync(self, image: Image.Image) -> List[Dict]:
        """
        Synchronous detection using Moondream SDK (runs in thread pool).
        
        Args:
            image: PIL Image to run detection on
            
        Returns:
            List of detection dictionaries
        """
        if self._shutdown:
            return []
        
        all_detections = []
        
        # Call SDK for each object type
        for object_type in self.detect_objects:
            try:
                logger.debug(f"🔍 Detecting '{object_type}' via Moondream SDK")
                
                # Call SDK's detect method
                result = self.model.detect(image, object_type)
                
                # Parse SDK response format
                # SDK returns: {"objects": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}, ...]}
                if "objects" in result and isinstance(result["objects"], list):
                    for obj in result["objects"]:
                        detection = self._parse_detection_bbox(obj, object_type)
                        if detection:
                            all_detections.append(detection)
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect '{object_type}': {e}")
                continue
        
        logger.debug(f"🔍 SDK returned {len(all_detections)} objects across {len(self.detect_objects)} types")
        return all_detections
    
    def _parse_detection_bbox(self, obj: Dict, object_type: str) -> Optional[Dict]:
        """
        Parse and format a detection result from the SDK.
        
        Args:
            obj: Detection object from SDK
            object_type: Type of object being detected
            
        Returns:
            Formatted detection dictionary, or None if below confidence threshold
        """
        confidence = obj.get("confidence", 1.0)
        
        # Filter by confidence threshold
        if confidence < self.conf_threshold:
            return None
        
        bbox = [
            obj.get("x_min", 0),
            obj.get("y_min", 0),
            obj.get("x_max", 0),
            obj.get("y_max", 0)
        ]
        
        return {
            "label": object_type,
            "bbox": bbox,
            "confidence": confidence
        }
    
    def _normalize_bbox_coordinates(self, bbox: List[float], width: int, height: int) -> tuple:
        """
        Normalize bounding box coordinates to pixel values.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] (may be normalized or pixel coords)
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            Tuple of (x1, y1, x2, y2) as pixel coordinates
        """
        if len(bbox) != 4:
            return (0, 0, 0, 0)
        
        x1, y1, x2, y2 = bbox
        
        # Check if normalized coordinates (between 0 and 1)
        if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
            # Convert to pixel coordinates
            return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
        else:
            # Already pixel coordinates
            return (int(x1), int(y1), int(x2), int(y2))
    
    async def _process_and_add_frame(self, frame: av.VideoFrame):
        """
        Callback for VideoForwarder - process frame and publish.
        
        This is the main processing pipeline:
        1. Convert frame to numpy array
        2. Run inference
        3. Store results for state() method
        4. Annotate frame if detection enabled
        5. Publish processed frame
        
        Args:
            frame: Input video frame from VideoForwarder
        """
        try:
            # Convert to numpy array
            frame_array = frame.to_ndarray(format="rgb24")
            
            # Run inference
            results = await self._run_inference(frame_array)
            
            # Store results for state() method and LLM access
            self._last_results = results
            self._last_frame_time = asyncio.get_event_loop().time()
            self._last_frame_pil = Image.fromarray(frame_array)
            
            # Annotate frame with detections
            if results.get("detections"):
                frame_array = self._annotate_detections(frame_array, results)
            
            # Convert back to av.VideoFrame and publish
            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)
            
        except Exception as e:
            logger.error(f"❌ Frame processing failed: {e}")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)
    
    def _annotate_detections(self, frame_array: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame_array: Input frame as numpy array
            results: Inference results containing detections
            
        Returns:
            Annotated frame as numpy array
        """
        annotated = frame_array.copy()
        
        detections = results.get("detections", [])
        if not detections:
            return annotated
        
        height, width = frame_array.shape[:2]
        
        for detection in detections:
            # Parse bounding box and normalize to pixel coordinates
            bbox = detection.get("bbox", [])
            x1, y1, x2, y2 = self._normalize_bbox_coordinates(bbox, width, height)
            
            # Skip invalid bounding boxes
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            
            # Get label and confidence
            label = detection.get("label", "object")
            conf = detection.get("confidence", 0.0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label_text = f"{label} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return annotated
    
    def _summarize_detections(self, detections: List[Dict]) -> str:
        """
        Create text summary of detections for LLM.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Human-readable summary string
        """
        if not detections:
            return "No objects detected"
        
        # Count occurrences of each label
        label_counts: Dict[str, int] = {}
        for det in detections:
            label = det.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Create summary
        parts = []
        for label, count in label_counts.items():
            if count == 1:
                parts.append(f"1 {label}")
            else:
                parts.append(f"{count} {label}s")
        
        return "Detected: " + ", ".join(parts)
    
    def state(self) -> dict:
        """
        Return latest detection results for LLM context.
        
        Returns:
            Dictionary containing:
            - last_frame_timestamp: When the frame was processed
            - last_image: PIL Image for LLM vision models
            - detections_summary: Human-readable detection summary
            - detections_count: Number of objects detected
        """
        if not hasattr(self, "_last_results"):
            return {}
        
        state_dict = {}
        
        # Add timestamp
        if hasattr(self, "_last_frame_time"):
            state_dict["last_frame_timestamp"] = self._last_frame_time
        
        # Add last image for LLM vision models
        if hasattr(self, "_last_frame_pil"):
            state_dict["last_image"] = self._last_frame_pil
        
        # Add detection results
        if "detections" in self._last_results:
            detections = self._last_results["detections"]
            state_dict["detections_summary"] = self._summarize_detections(detections)
            state_dict["detections_count"] = len(detections)
        
        return state_dict
    
    def close(self):
        """Clean up resources."""
        self._shutdown = True
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
        logger.info("🛑 Moondream Processor closed")

