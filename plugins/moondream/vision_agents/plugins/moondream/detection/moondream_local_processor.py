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
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.warmup import Warmable
from vision_agents.plugins.moondream.detection.moondream_video_track import (
    MoondreamVideoTrack,
)
from vision_agents.plugins.moondream.moondream_utils import (
    annotate_detections,
    handle_device,
    parse_detection_bbox,
)

logger = logging.getLogger(__name__)


class LocalDetectionProcessor(VideoProcessorPublisher, Warmable):
    """Performs real-time object detection on video streams using local Moondream 3 model.

    This processor downloads and runs the moondream3-preview model locally from Hugging Face,
    providing the same functionality as the cloud API version without requiring an API key.

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
        fps: Frame processing rate (default: 30)
        max_workers: Number of worker threads for CPU-intensive operations (default: 10)
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
        fps: int = 30,
        max_workers: int = 10,
        force_cpu: bool = False,
        model_name: str = "moondream/moondream3-preview",
        options: Optional[AgentOptions] = None,
    ):
        if options is None:
            self.options = default_agent_options()
        else:
            self.options = options
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.fps = fps
        self.max_workers = max_workers
        self._shutdown = False

        if force_cpu:
            self._device, self._dtype = torch.device("cpu"), torch.float32
        else:
            self._device, self._dtype = handle_device()

        self._last_results: Dict[str, Any] = {}
        self._last_frame_time: Optional[float] = None
        self._last_frame_pil: Optional[Image.Image] = None

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

        # Video track for publishing (if used as video publisher)
        self._video_track: MoondreamVideoTrack = MoondreamVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # Model will be loaded in start() method
        self.model = None

        logger.info("🌙 Moondream Local Processor initialized")
        logger.info(f"🎯 Detection configured for objects: {self.detect_objects}")
        logger.info(f"🔧 Device: {self.device}")

    @property
    def device(self) -> str:
        """Return the device type as a string (e.g., 'cuda', 'cpu')."""
        return str(self._device)

    async def on_warmup(self):
        logger.info(f"Loading Moondream model: {self.model_name}")
        logger.info(f"Device: {self._device}")

        # Load model in thread pool to avoid blocking event loop
        # Transformers handles downloading and caching automatically via Hugging Face Hub
        model = await asyncio.to_thread(  # type: ignore[func-returns-value]
            lambda: self._load_model_sync()
        )
        logger.info("✅ Moondream model loaded")
        return model

    def on_warmed_up(self, model) -> None:
        self.model = model

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
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
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
            raise ValueError("The Moondream model is not loaded")

        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Moondream subscribing to shared VideoForwarder at {self.fps} FPS"
            )
            self._video_forwarder.add_frame_handler(
                self._process_and_add_frame, fps=float(self.fps), name="moondream_local"
            )
        else:
            self._video_forwarder = VideoForwarder(
                incoming_track,  # type: ignore[arg-type]
                max_buffer=30,  # 1 second at 30fps
                fps=self.fps,
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
        try:
            frame_array = frame.to_ndarray(format="rgb24")
            results = await self._run_inference(frame_array)

            self._last_results = results
            self._last_frame_time = asyncio.get_event_loop().time()
            self._last_frame_pil = Image.fromarray(frame_array)

            if results.get("detections"):
                frame_array = annotate_detections(
                    frame_array,
                    results,
                    font=self._font,
                    font_scale=self._font_scale,
                    font_thickness=self._font_thickness,
                    bbox_color=self._bbox_color,
                    text_color=self._text_color,
                )

            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            await self._video_track.add_frame(frame)

    async def close(self):
        """Clean up resources."""
        self._shutdown = True
        self.executor.shutdown(wait=False)
        if self.model is not None:
            # Clear model reference to free memory
            del self.model
            self.model = None
        logger.info("🛑 Moondream Local Processor closed")
