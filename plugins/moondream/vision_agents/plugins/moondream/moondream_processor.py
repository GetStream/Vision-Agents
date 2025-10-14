import asyncio
import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import aiohttp
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
    """
    Moondream 3 vision processor.
    
    Supports object detection, VQA, counting, and captioning with multiple inference modes.
    
    Args:
        mode: Inference mode ("cloud", "local", or "fal")
        skills: List of skills to enable (detection, vqa, counting, caption)
        api_key: API key for cloud/FAL modes
        model_path: Path to local model (for local mode)
        conf_threshold: Confidence threshold for detections
        vqa_prompt: Default question for VQA skill
        fps: Frame processing rate
        interval: Processing interval in seconds
        max_workers: Number of worker threads
        device: Device for local inference ("cpu" or "cuda")
    """
    
    def __init__(
        self,
        mode: str = "cloud",
        skills: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        vqa_prompt: str = "What do you see?",
        fps: int = 30,
        interval: int = 0,
        max_workers: int = 4,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)
        
        self.mode = mode
        self.skills = skills or ["detection"]
        self.api_key = api_key
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.vqa_prompt = vqa_prompt
        self.fps = fps
        self.device = device
        self.max_workers = max_workers
        self._shutdown = False
        
        # Thread pool for CPU-intensive inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="moondream_processor"
        )
        
        # Video track for publishing (if used as video publisher)
        self._video_track: MoondreamVideoTrack = MoondreamVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None
        
        # Initialize model for local mode
        if mode == "local":
            self._load_local_model()
        
        logger.info(f"🌙 Moondream Processor initialized with mode: {mode}, skills: {self.skills}")
    
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
    
    def _load_local_model(self):
        """Load Moondream local model."""
        try:
            # Import moondream SDK
            # from moondream import Moondream
            # self.model = Moondream(self.model_path, device=self.device)
            logger.info(f"✅ Local model loading not yet implemented")
        except Exception as e:
            logger.warning(f"⚠️ Could not load local model: {e}")
            self.model = None
    
    async def run_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """
        Run inference based on configured mode.
        
        Args:
            frame_array: Input frame as numpy array (RGB format)
            
        Returns:
            Dictionary containing results for all enabled skills
        """
        if self.mode == "cloud":
            return await self._cloud_inference(frame_array)
        elif self.mode == "local":
            return await self._local_inference(frame_array)
        elif self.mode == "fal":
            return await self._fal_inference(frame_array)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    async def _cloud_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """
        Call Moondream Cloud API.
        
        Args:
            frame_array: Input frame as numpy array
            
        Returns:
            Dictionary with inference results
        """
        try:
            # Convert frame to PIL Image
            image = Image.fromarray(frame_array)
            
            # Encode image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            results = {}
            
            # Run inference for each enabled skill
            async with aiohttp.ClientSession() as session:
                for skill in self.skills:
                    if skill == "detection":
                        result = await self._call_detection_api(session, img_base64)
                        results["detections"] = result
                    elif skill == "vqa":
                        result = await self._call_vqa_api(session, img_base64, self.vqa_prompt)
                        results["vqa_response"] = result
                    elif skill == "counting":
                        result = await self._call_counting_api(session, img_base64)
                        results["count"] = result
                    elif skill == "caption":
                        result = await self._call_caption_api(session, img_base64)
                        results["caption"] = result
            
            return results
        except Exception as e:
            logger.error(f"❌ Cloud inference failed: {e}")
            return {}
    
    async def _call_detection_api(self, session: aiohttp.ClientSession, img_base64: str) -> List[Dict]:
        """Call Moondream detection API."""
        # Placeholder - actual API endpoint needed
        logger.debug("🔍 Calling detection API (mock)")
        return []
    
    async def _call_vqa_api(self, session: aiohttp.ClientSession, img_base64: str, question: str) -> str:
        """Call Moondream VQA API."""
        # Placeholder - actual API endpoint needed
        logger.debug(f"❓ Calling VQA API with question: {question} (mock)")
        return ""
    
    async def _call_counting_api(self, session: aiohttp.ClientSession, img_base64: str) -> int:
        """Call Moondream counting API."""
        # Placeholder - actual API endpoint needed
        logger.debug("🔢 Calling counting API (mock)")
        return 0
    
    async def _call_caption_api(self, session: aiohttp.ClientSession, img_base64: str) -> str:
        """Call Moondream caption API."""
        # Placeholder - actual API endpoint needed
        logger.debug("📝 Calling caption API (mock)")
        return ""
    
    async def _local_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """
        Run local Moondream model inference.
        
        Args:
            frame_array: Input frame as numpy array
            
        Returns:
            Dictionary with inference results
        """
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self.executor, self._local_inference_sync, frame_array
                ),
                timeout=5.0
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("⏰ Local inference TIMEOUT")
            return {}
        except Exception as e:
            logger.error(f"❌ Local inference error: {e}")
            return {}
    
    def _local_inference_sync(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """
        Synchronous local inference (runs in thread pool).
        
        Args:
            frame_array: Input frame as numpy array
            
        Returns:
            Dictionary with inference results
        """
        if self._shutdown:
            return {}
        
        # Placeholder - actual local model inference
        logger.debug("🤖 Running local inference (not yet implemented)")
        return {}
    
    async def _fal_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """
        Call FAL.ai Moondream API.
        
        Args:
            frame_array: Input frame as numpy array
            
        Returns:
            Dictionary with inference results
        """
        try:
            # Convert frame to PIL Image
            image = Image.fromarray(frame_array)
            
            # Encode image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Placeholder - actual FAL.ai API call
            logger.debug("🚀 Calling FAL.ai API (not yet implemented)")
            return {}
        except Exception as e:
            logger.error(f"❌ FAL inference failed: {e}")
            return {}
    
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
            results = await self.run_inference(frame_array)
            
            # Store results for state() method and LLM access
            self._last_results = results
            self._last_frame_time = asyncio.get_event_loop().time()
            self._last_frame_pil = Image.fromarray(frame_array)
            
            # Annotate frame if detection skill enabled
            if "detection" in self.skills and results.get("detections"):
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
            # Parse bounding box
            # Moondream might return normalized coords [0,1] or pixel coords
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Check if normalized coordinates (between 0 and 1)
            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                # Convert to pixel coordinates
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            else:
                # Already pixel coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
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
        Return latest inference results for LLM context.
        
        This enables LLM to access visual understanding including:
        - VQA responses (answers to visual questions)
        - Counting results
        - Image captions
        - Detection summaries
        - Last frame as PIL Image
        
        Returns:
            Dictionary containing latest inference results
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
        
        # Add skill-specific results
        if "vqa_response" in self._last_results:
            state_dict["vqa_response"] = self._last_results["vqa_response"]
        
        if "count" in self._last_results:
            state_dict["count"] = self._last_results["count"]
        
        if "caption" in self._last_results:
            state_dict["caption"] = self._last_results["caption"]
        
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

