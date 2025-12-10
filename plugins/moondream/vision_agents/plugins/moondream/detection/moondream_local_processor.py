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
from PIL import Image
from transformers import AutoModelForCausalLM

from vision_agents.core.agents.agent_types import AgentOptions, default_agent_options
from vision_agents.core.processors.base_processor import (
    VideoProcessorMixin,
    VideoPublisherMixin,
    AudioVideoProcessor,
)
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.plugins.moondream.moondream_utils import (
    parse_detection_bbox,
    annotate_detections,
    handle_device,
)
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)


class LocalDetectionProcessor(
    AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin
):
    """Performs real-time object detection on video streams using local Moondream 3 model.

    This processor downloads and runs the moondream3-preview model locally from Hugging Face,
    providing the same functionality as the cloud API version without requiring an API key.

    Detection runs asynchronously in the background while frames pass through at
    full FPS. The last known detection results are overlaid on each frame.

    Note: The moondream3-preview model is gated and requires authentication:
    - Request access at https://huggingface.co/moondream/moondream3-preview
    - Once approved, authenticate using one of:
      - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here
      - Run: huggingface-cli login

    Args:
        conf_threshold: Confidence threshold for detections (default: 0.3)
        detect_objects: Object(s) to detect. Moondream uses zero-shot detection,
                       so any object string works. Examples: "person", "car",
                       "basketball", ["person", "car", "dog"]. Default: "person"
        detection_fps: Rate at which to run detection (default: 10.0).
                      Lower values reduce CPU/GPU load while maintaining smooth video.
        interval: Processing interval in seconds (default: 0)
        max_workers: Number of worker threads for CPU-intensive operations (default: 2)
        force_cpu: If True, force CPU usage even if CUDA/MPS is available (default: False).
                  Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU. We recommend running on CUDA for best performance.
        model_name: Hugging Face model identifier (default: "moondream/moondream3-preview")
        options: AgentOptions for model directory configuration. If not provided,
                 uses default_agent_options() which defaults to tempfile.gettempdir()
    """

    name = "moondream_local"

    def __init__(
        self,
        conf_threshold: float = 0.3,
        detect_objects: Union[str, List[str]] = "person",
        detection_fps: float = 10.0,
        interval: int = 0,
        max_workers: int = 2,
        force_cpu: bool = False,
        model_name: str = "moondream/moondream3-preview",
        options: Optional[AgentOptions] = None,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)

        if options is None:
            self.options = default_agent_options()
        else:
            self.options = options
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.detection_fps = detection_fps
        self.max_workers = max_workers
        self._shutdown = False

        if force_cpu:
            self._device, self._dtype = torch.device("cpu"), torch.float32
        else:
            self._device, self._dtype = handle_device()

        # Parallel detection state - track when results were requested to handle out-of-order completion
        self._last_detection_time: float = 0.0
        self._last_result_time: float = 0.0
        self._cached_results: Dict[str, Any] = {"detections": []}

        # Font configuration constants for drawing efficiency
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 2
        self._bbox_color = (0, 255, 0)
        self._text_color = (0, 0, 0)

        # Normalize detect_objects to list
        self.detect_objects = (
            [detect_objects]
            if isinstance(detect_objects, str)
            else list(detect_objects)
        )

        # Thread pool for CPU-intensive inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="moondream_local_processor"
        )

        # Video track for publishing at 30 FPS with minimal buffering
        self._video_track: QueuedVideoTrack = QueuedVideoTrack(fps=30, max_queue_size=5)
        self._video_forwarder: Optional[VideoForwarder] = None

        # Model will be loaded in start() method
        self.model = None

        logger.info("🌙 Moondream Local Processor initialized")
        logger.info(f"🎯 Detection configured for objects: {self.detect_objects}")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📹 Detection FPS: {detection_fps}")

    @property
    def device(self) -> str:
        """Return the device type as a string (e.g., 'cuda', 'cpu')."""
        return str(self._device)

    async def warmup(self):
        # Prepare model asynchronously
        await self._prepare_moondream()

    async def _prepare_moondream(self):
        """Load the Moondream model from Hugging Face."""
        logger.info(f"Loading Moondream model: {self.model_name}")
        logger.info(f"Device: {self._device}")

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

            load_kwargs: Dict[str, Any] = {}
            # Add token if available (transformers will use env var automatically, but explicit is clearer)
            if hf_token:
                load_kwargs["token"] = hf_token
            else:
                # Use True to let transformers try to read from environment or cached login
                load_kwargs["token"] = True  # type: ignore[assignment]

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": self._device},
                dtype=self._dtype,
                trust_remote_code=True,
                cache_dir=self.options.model_dir,
                **load_kwargs,
            ).to(self._device)  # type: ignore[arg-type]

            model.eval()
            logger.info(f"✅ Model loaded on {self._device} device")

            # Compile model for fast inference
            try:
                model.compile()
            except Exception as compile_error:
                # If compilation fails, log and continue without compilation
                logger.warning(
                    f"⚠️ Model compilation failed, continuing without compilation: {compile_error}"
                )

            return model
        except Exception as e:
            error_msg = str(e)
            if (
                "gated repo" in error_msg.lower()
                or "403" in error_msg
                or "authorized" in error_msg.lower()
            ):
                logger.exception(
                    "❌ Failed to load Moondream model: Model requires authentication.\n"
                    "This model is gated and requires access approval:\n"
                    f"1. Visit https://huggingface.co/{self.model_name} to request access\n"
                    "2. Once approved, authenticate using one of:\n"
                    "   - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here\n"
                    "   - Run: huggingface-cli login\n"
                    f"Original error: {e}"
                )
            else:
                logger.exception(f"❌ Failed to load Moondream model: {e}")
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
            # Use the shared forwarder at its native FPS
            self._video_forwarder = shared_forwarder
            logger.info("🎥 Moondream subscribing to shared VideoForwarder")
            self._video_forwarder.add_frame_handler(
                self._process_and_add_frame, name="moondream_local"
            )
        else:
            # Create our own VideoForwarder at default FPS with minimal buffering
            self._video_forwarder = VideoForwarder(
                incoming_track,  # type: ignore[arg-type]
                max_buffer=5,
                name="moondream_local_forwarder",
            )

            # Add frame handler (starts automatically)
            self._video_forwarder.add_frame_handler(self._process_and_add_frame)

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
            logger.exception(f"❌ Local inference failed: {e}")
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
                        detection = parse_detection_bbox(
                            obj, object_type, self.conf_threshold
                        )
                        if detection:
                            all_detections.append(detection)

            except Exception as e:
                logger.warning(f"⚠️ Failed to detect '{object_type}': {e}")
                continue

        logger.debug(
            f"🔍 Model returned {len(all_detections)} objects across {len(self.detect_objects)} types"
        )
        return all_detections

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        """Process frame: pass through immediately, run detection asynchronously."""
        try:
            frame_array = frame.to_ndarray(format="rgb24")
            now = asyncio.get_event_loop().time()

            # Check if we should start a new detection based on detection_fps
            detection_interval = (
                1.0 / self.detection_fps if self.detection_fps > 0 else float("inf")
            )
            should_detect = (now - self._last_detection_time) >= detection_interval

            if should_detect:
                # Start detection in background (don't await) - runs in parallel
                self._last_detection_time = now
                asyncio.create_task(
                    self._run_detection_background(frame_array.copy(), now)
                )

            # Annotate frame with cached detections
            if self._cached_results.get("detections"):
                frame_array = annotate_detections(
                    frame_array,
                    self._cached_results,
                    font=self._font,
                    font_scale=self._font_scale,
                    font_thickness=self._font_thickness,
                    bbox_color=self._bbox_color,
                    text_color=self._text_color,
                )

            # Convert back to av.VideoFrame and publish immediately
            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            await self._video_track.add_frame(frame)

    async def _run_detection_background(
        self, frame_array: np.ndarray, request_time: float
    ):
        """Run detection in background and update cached results if newer."""
        try:
            results = await self._run_inference(frame_array)
            # Only update cache if this result is newer than current cached result
            if request_time > self._last_result_time:
                self._cached_results = results
                self._last_result_time = request_time
                logger.debug(
                    f"🔍 Detection complete: {len(results.get('detections', []))} objects"
                )
            else:
                logger.debug("🔍 Detection complete but discarded (newer result exists)")
        except Exception as e:
            logger.warning(f"⚠️ Background detection failed: {e}")

    def close(self):
        """Clean up resources."""
        self._shutdown = True
        self.executor.shutdown(wait=False)
        if self.model is not None:
            # Clear model reference to free memory
            del self.model
            self.model = None
        logger.info("🛑 Moondream Local Processor closed")
