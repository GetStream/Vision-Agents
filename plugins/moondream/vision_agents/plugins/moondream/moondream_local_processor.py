import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import aiortc
import av
import cv2
import numpy as np
import torch
from aiortc import VideoStreamTrack
from PIL import Image
from transformers import AutoModelForCausalLM

from vision_agents.core.agents.agents import AgentOptions, default_agent_options
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


class LocalDetectionProcessor(AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin):
    """Performs real-time object detection on video streams using local Moondream 3 model.
    
    This processor downloads and runs the moondream3-preview model locally from Hugging Face,
    providing the same functionality as the cloud API version without requiring an API key.
    
    Note: The moondream3-preview model is gated and requires authentication:
    - Request access at https://huggingface.co/moondream/moondream3-preview
    - Once approved, authenticate using one of:
      - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here
      - Run: huggingface-cli login
    
    Args:
        conf_threshold: Confidence threshold for detections
        detect_objects: Object(s) to detect. Moondream uses zero-shot detection,
                       so any object string works. Examples: "person", "car",
                       "basketball", ["person", "car", "dog"]. Default: "person"
        fps: Frame processing rate
        interval: Processing interval in seconds
        max_workers: Number of worker threads
        device: Device to run inference on ('cuda', 'mps', or 'cpu'). 
               Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU.
        model_name: Hugging Face model identifier (default: "moondream/moondream3-preview")
        options: AgentOptions for model directory configuration. If not provided,
                 uses default_agent_options() which defaults to tempfile.gettempdir()
    """
    
    def __init__(
        self,
        conf_threshold: float = 0.3,
        detect_objects: Union[str, List[str]] = "person",
        fps: int = 30,
        interval: int = 0,
        max_workers: int = 10,
        device: Optional[str] = None,
        model_name: str = "moondream/moondream3-preview",
        options: Optional[AgentOptions] = None,
        *args,
        **kwargs,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)
        
        if options is None:
            self.options = default_agent_options()
        else:
            self.options = options
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.max_workers = max_workers
        self._shutdown = False
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Moondream model has CUDA dependencies that don't work on MPS
                # Use CPU instead to avoid runtime errors
                self.device = "cpu"
                logger.info("⚠️ MPS detected but using CPU (moondream model has CUDA dependencies incompatible with MPS)")
            else:
                self.device = "cpu"
        else:
            # Override MPS to CPU if explicitly set (moondream doesn't work with MPS)
            if device == "mps":
                self.device = "cpu"
                logger.warning("⚠️ MPS device requested but using CPU instead (moondream model has CUDA dependencies incompatible with MPS)")
            else:
                self.device = device
        
        # Initialize state tracking attributes
        self._last_results = {}
        self._last_frame_time = None
        self._last_frame_pil = None
        
        # Font configuration constants for drawing efficiency
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 2
        self._bbox_color = (0, 255, 0)
        self._text_color = (0, 0, 0)
        
        # Normalize detect_objects to list
        self.detect_objects = [detect_objects] if isinstance(detect_objects, str) else list(detect_objects)
        
        # Thread pool for CPU-intensive inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="moondream_local_processor"
        )
        
        # Video track for publishing (if used as video publisher)
        self._video_track: MoondreamVideoTrack = MoondreamVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None
        
        # Model will be loaded in start() method
        self.model = None
        
        logger.info("🌙 Moondream Local Processor initialized")
        logger.info(f"🎯 Detection configured for objects: {self.detect_objects}")
        logger.info(f"🔧 Device: {self.device}")
    
    async def start(self):
        """Initialize and load the model."""
        # Ensure model directory exists
        os.makedirs(self.options.model_dir, exist_ok=True)
        
        # Prepare model asynchronously
        await self._prepare_moondream()
    
    async def _prepare_moondream(self):
        """Load the Moondream model from Hugging Face."""
        logger.info(f"Loading Moondream model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load model in thread pool to avoid blocking event loop
        # Transformers handles downloading and caching automatically via Hugging Face Hub
        self.model = await asyncio.to_thread(  # type: ignore[func-returns-value]
            lambda: self._load_model_sync()
        )
        logger.info("✅ Moondream model loaded")
    
    def _load_model_sync(self):
        """Synchronous model loading function run in thread pool."""
        try:
            # Check for Hugging Face token (required for gated models)
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.warning(
                    "⚠️ HF_TOKEN environment variable not set. "
                    "This model requires authentication. "
                    "Set HF_TOKEN or run 'huggingface-cli login'"
                )
            
            load_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                "cache_dir": self.options.model_dir,  # Use agent's model directory for caching
            }
            
            # Add token if available (transformers will use env var automatically, but explicit is clearer)
            if hf_token:
                load_kwargs["token"] = hf_token
            else:
                # Use True to let transformers try to read from environment or cached login
                load_kwargs["token"] = True
            
            # Handle device placement based on device type
            if self.device == "cuda":
                # CUDA: Use device_map for efficient multi-GPU support
                load_kwargs["device_map"] = {"": "cuda"}
            else:
                # CPU: load directly on CPU (MPS is automatically converted to CPU in __init__)
                load_kwargs["device_map"] = "cpu"
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            
            # Ensure model is in eval mode for inference
            model.eval()
            
            if self.device == "cuda":
                logger.info("✅ Model loaded on CUDA device")
            else:
                logger.info("✅ Model loaded on CPU device")
            
            # Compile model for fast inference (as per HF documentation)
            try:
                model.compile()
            except Exception as compile_error:
                # If compilation fails, log and continue without compilation
                logger.warning(f"⚠️ Model compilation failed, continuing without compilation: {compile_error}")
            
            return model
        except ImportError as e:
            raise ImportError(
                "transformers library is not installed. "
                "Please install it: pip install transformers torch"
            ) from e
        except Exception as e:
            error_msg = str(e)
            if "gated repo" in error_msg.lower() or "403" in error_msg or "authorized" in error_msg.lower():
                logger.error(
                    "❌ Failed to load Moondream model: Model requires authentication.\n"
                    "This model is gated and requires access approval:\n"
                    f"1. Visit https://huggingface.co/{self.model_name} to request access\n"
                    "2. Once approved, authenticate using one of:\n"
                    "   - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here\n"
                    "   - Run: huggingface-cli login\n"
                    f"Original error: {e}"
                )
            else:
                logger.error(f"❌ Failed to load Moondream model: {e}")
            raise
    
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
        """
        logger.info("✅ Moondream process_video starting")
        
        # Ensure model is loaded
        if self.model is None:
            await self._prepare_moondream()
        
        if shared_forwarder is not None:
            # Use the shared forwarder
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Moondream subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            await self._video_forwarder.start_event_consumer(
                self._process_and_add_frame,
                fps=float(self.fps),
                consumer_name="moondream_local"
            )
        else:
            # Create our own VideoForwarder
            self._video_forwarder = VideoForwarder(
                incoming_track,  # type: ignore[arg-type]
                max_buffer=30,  # 1 second at 30fps
                fps=self.fps,
                name="moondream_local_forwarder",
            )
            
            # Start the forwarder
            await self._video_forwarder.start()
            await self._video_forwarder.start_event_consumer(
                self._process_and_add_frame
            )
            
        logger.info("✅ Moondream video processing pipeline started")
    
    def publish_video_track(self):
        logger.info("📹 publish_video_track called")
        return self._video_track
    
    async def _run_inference(self, frame_array: np.ndarray) -> Dict[str, Any]:
        try:
            # Convert frame to PIL Image
            image = Image.fromarray(frame_array)
            
            # Call model for each object type
            # The model's detect() is synchronous, so wrap in executor
            loop = asyncio.get_event_loop()
            all_detections = await loop.run_in_executor(
                self.executor, self._run_detection_sync, image
            )
            
            return {"detections": all_detections}
        except Exception as e:
            logger.error(f"❌ Local inference failed: {e}")
            return {}
    
    def _run_detection_sync(self, image: Image.Image) -> List[Dict]:
        if self._shutdown or self.model is None:
            return []
        
        all_detections = []
        
        # Call model for each object type
        for object_type in self.detect_objects:
            try:
                logger.debug(f"🔍 Detecting '{object_type}' via Moondream model")
                
                # Call model's detect method
                result = self.model.detect(image, object_type)
                
                # Parse model response format
                # Model returns: {"objects": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}, ...]}
                if "objects" in result:
                    for obj in result["objects"]:
                        detection = self._parse_detection_bbox(obj, object_type)
                        if detection:
                            all_detections.append(detection)
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect '{object_type}': {e}")
                continue
        
        logger.debug(f"🔍 Model returned {len(all_detections)} objects across {len(self.detect_objects)} types")
        return all_detections
    
    def _parse_detection_bbox(self, obj: Dict, object_type: str) -> Optional[Dict]:
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
        annotated = frame_array.copy()
        
        detections = results.get("detections", [])
        if not detections:
            return annotated
        
        height, width = frame_array.shape[:2]
        
        # Pre-calculate baseline text metrics once per frame for efficiency
        sample_text = "object 0.00"  # Representative text for baseline calculation
        (_, text_height), baseline = cv2.getTextSize(
            sample_text, self._font, self._font_scale, self._font_thickness
        )
        
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
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self._bbox_color, 2)
            
            # Draw label background
            label_text = f"{label} {conf:.2f}"
            # Calculate text width for this specific label (varies by content)
            (text_width, _), _ = cv2.getTextSize(
                label_text, self._font, self._font_scale, self._font_thickness
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                self._bbox_color,
                -1
            )
            
            # Draw label text using cached parameters
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - baseline - 5),
                self._font,
                self._font_scale,
                self._text_color,
                self._font_thickness
            )
        
        return annotated
    
    def _summarize_detections(self, detections: List[Dict]) -> str:
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
        if not self._last_results:
            return {}
        
        state_dict = {}
        
        # Add timestamp
        if self._last_frame_time is not None:
            state_dict["last_frame_timestamp"] = self._last_frame_time
        
        # Add last image for LLM vision models
        if self._last_frame_pil is not None:
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
        self.executor.shutdown(wait=False)
        if self.model is not None:
            # Clear model reference to free memory
            del self.model
            self.model = None
        logger.info("🛑 Moondream Local Processor closed")


