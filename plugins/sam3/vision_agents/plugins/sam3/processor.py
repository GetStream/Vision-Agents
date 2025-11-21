import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import aiortc
import av
import cv2
import numpy as np
import torch
from PIL import Image

from vision_agents.core.agents.agent_types import AgentOptions, default_agent_options
from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    VideoProcessorMixin,
    VideoPublisherMixin,
)
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.plugins.sam3.video_track import Sam3VideoTrack

logger = logging.getLogger(__name__)


class VideoSegmentationProcessor(AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin):
    """Performs real-time video segmentation using SAM 3 (Segment Anything with Concepts).
    
    This processor uses Meta's SAM 3 model via Hugging Face Transformers to segment 
    objects in video streams based on text prompts. The segmentation prompt can be 
    changed dynamically via function calls.
    
    SAM 3 performs Promptable Concept Segmentation (PCS), finding ALL instances of 
    objects matching the text prompt in each frame.
    
    Note: SAM 3 requires authentication:
    - Request access at https://huggingface.co/facebook/sam3
    - Once approved, authenticate using one of:
      - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here
      - Run: huggingface-cli login
    
    Args:
        text_prompt: Initial text prompt for segmentation (default: "person")
        threshold: Confidence threshold for detections (default: 0.5)
        mask_threshold: Threshold for mask binarization (default: 0.5)
        fps: Frame processing rate (default: 30)
        interval: Processing interval in seconds (default: 0)
        max_workers: Number of worker threads for CPU-intensive operations (default: 10)
        force_cpu: If True, force CPU usage even if CUDA is available (default: False)
        model_id: Hugging Face model ID (default: "facebook/sam3")
        options: AgentOptions for model directory configuration
    
    Function Calls:
        The processor supports these function calls from the AI:
        - change_prompt: Change the segmentation prompt dynamically
            Args:
                prompt (str): New text prompt for segmentation
    """
    name = "sam3_video_segmentation"

    def __init__(
        self,
        text_prompt: str = "person",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        fps: int = 30,
        interval: int = 0,
        max_workers: int = 10,
        force_cpu: bool = False,
        model_id: str = "facebook/sam3",
        options: Optional[AgentOptions] = None,
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)

        if options is None:
            self.options = default_agent_options()
        else:
            self.options = options

        self.text_prompt = text_prompt
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.fps = fps
        self.max_workers = max_workers
        self.model_id = model_id
        self._shutdown = False

        if force_cpu:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._last_results: Dict[str, Any] = {}
        self._last_frame_time: Optional[float] = None

        # Font configuration for drawing
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 2

        # Thread pool for CPU-intensive inference
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="sam3_processor"
        )

        # Video track for publishing
        self._video_track: Sam3VideoTrack = Sam3VideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        # SAM3 model and processor will be loaded in warmup()
        self.model = None
        self.processor = None

        logger.info("🎯 SAM3 Video Segmentation Processor initialized")
        logger.info(f"📝 Initial prompt: '{self.text_prompt}'")
        logger.info(f"🔧 Device: {self.device}")

    @property
    def device(self) -> str:
        """Return the device type as a string."""
        return str(self._device)

    async def warmup(self):
        """Prepare the SAM3 model asynchronously."""
        await self._prepare_sam3()

    async def _prepare_sam3(self):
        """Load the SAM3 model and processor from Hugging Face."""
        logger.info(f"Loading SAM3 model: {self.model_id}")
        logger.info(f"Device: {self._device}")

        try:
            # Load model in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            self.model, self.processor = await loop.run_in_executor(
                self.executor, self._load_sam3_sync
            )
            logger.info("✅ SAM3 model loaded successfully")

        except Exception as e:
            logger.exception(f"❌ Failed to load SAM3 model: {e}")
            raise

    def _load_sam3_sync(self):
        """Synchronous SAM3 model loading function."""
        try:
            # Import transformers here to avoid import errors if not installed
            from transformers import Sam3Model, Sam3Processor

            # Load model and processor from Hugging Face
            logger.info("Downloading SAM3 model from Hugging Face...")
            model = Sam3Model.from_pretrained(
                self.model_id,
                cache_dir=self.options.model_dir,
            ).to(self._device)
            
            processor = Sam3Processor.from_pretrained(
                self.model_id,
                cache_dir=self.options.model_dir,
            )
            
            model.eval()
            logger.info(f"✅ Model loaded on {self._device} device")
            
            return model, processor

        except ImportError as e:
            logger.exception(
                "❌ Failed to import transformers. Please install it:\n"
                "pip install transformers>=4.40.0\n"
                f"Error: {e}"
            )
            raise
        except Exception as e:
            error_msg = str(e)
            if "gated repo" in error_msg.lower() or "403" in error_msg or "authorized" in error_msg.lower():
                logger.exception(
                    "❌ Failed to load SAM3 model: Model requires authentication.\n"
                    "This model is gated and requires access approval:\n"
                    f"1. Visit https://huggingface.co/{self.model_id} to request access\n"
                    "2. Once approved, authenticate using one of:\n"
                    "   - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here\n"
                    "   - Run: huggingface-cli login\n"
                    f"Original error: {e}"
                )
            else:
                logger.exception(f"❌ Failed to load SAM3 model: {e}")
            raise

    async def process_video(
        self,
        incoming_track: aiortc.mediastreams.MediaStreamTrack,
        participant: Any,
        shared_forwarder: VideoForwarder,
    ):
        """
        Process incoming video track.
        
        Sets up the video processing pipeline with SAM3 segmentation.
        """
        logger.info("✅ SAM3 process_video starting")

        # Ensure model is loaded
        if self.model is None or self.processor is None:
            await self._prepare_sam3()

        # Use the shared forwarder
        self._video_forwarder = shared_forwarder
        logger.info(
            f"🎥 SAM3 subscribing to shared VideoForwarder at {self.fps} FPS"
        )
        self._video_forwarder.add_frame_handler(
            self._process_and_add_frame,
            fps=float(self.fps),
            name="sam3_segmentation"
        )

        logger.info("✅ SAM3 video processing pipeline started")

    def publish_video_track(self):
        """Return the video track for publishing annotated frames."""
        logger.info("📹 publish_video_track called")
        return self._video_track

    async def change_prompt(self, prompt: str) -> Dict[str, str]:
        """
        Function call: Change the segmentation prompt dynamically.
        
        Args:
            prompt: New text prompt for segmentation
            
        Returns:
            Status message indicating the prompt was changed
        """
        old_prompt = self.text_prompt
        self.text_prompt = prompt
        logger.info(f"🔄 Segmentation prompt changed: '{old_prompt}' → '{prompt}'")
        
        return {
            "status": "success",
            "message": f"Segmentation prompt changed from '{old_prompt}' to '{prompt}'",
            "new_prompt": prompt
        }

    async def _run_segmentation(self, frame_array: np.ndarray) -> Dict[str, Any]:
        """Run SAM3 segmentation on a frame."""
        try:
            # Convert frame to PIL Image
            image = Image.fromarray(frame_array)

            # Run segmentation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._segment_frame_sync, image
            )

            return result
        except Exception as e:
            logger.exception(f"❌ Segmentation failed: {e}")
            return {}

    def _segment_frame_sync(self, image: Image.Image) -> Dict[str, Any]:
        """Synchronous segmentation function using Transformers API."""
        if self._shutdown or self.model is None or self.processor is None:
            return {}

        try:
            logger.debug(f"🔍 Segmenting with prompt: '{self.text_prompt}'")
            
            # Prepare inputs with text prompt
            inputs = self.processor(
                images=image,
                text=self.text_prompt,
                return_tensors="pt"
            ).to(self._device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process to get masks, boxes, and scores
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            # Convert to numpy for easier processing
            result_dict = {
                "masks": results["masks"].cpu().numpy() if len(results["masks"]) > 0 else np.array([]),
                "boxes": results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([]),
                "scores": results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([]),
            }

            logger.debug(f"✅ Found {len(result_dict['masks'])} objects matching '{self.text_prompt}'")
            return result_dict

        except Exception as e:
            logger.warning(f"⚠️ Frame segmentation failed: {e}")
            return {}

    async def _process_and_add_frame(self, frame: av.VideoFrame):
        """Process a frame with SAM3 segmentation and add to output."""
        try:
            frame_array = frame.to_ndarray(format="rgb24")
            
            # Run segmentation
            results = await self._run_segmentation(frame_array)

            self._last_results = results
            self._last_frame_time = asyncio.get_event_loop().time()

            # Annotate frame with segmentation masks
            if results and len(results.get("masks", [])) > 0:
                frame_array = self._annotate_segmentation(frame_array, results)

            # Convert back to video frame
            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)

        except Exception as e:
            logger.exception(f"❌ Frame processing failed: {e}")
            # Pass through original frame on error
            await self._video_track.add_frame(frame)

    def _annotate_segmentation(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw segmentation masks and labels on frame."""
        try:
            masks = results.get("masks", [])
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])

            # Draw masks with transparency
            if len(masks) > 0:
                for i, mask in enumerate(masks):
                    # Create colored overlay for mask
                    # Use different colors for multiple objects
                    colors = [
                        (0, 255, 0),    # Green
                        (255, 0, 0),    # Blue
                        (0, 255, 255),  # Yellow
                        (255, 0, 255),  # Magenta
                        (255, 255, 0),  # Cyan
                    ]
                    color = colors[i % len(colors)]
                    
                    overlay = frame.copy()
                    # Mask is already 2D binary mask
                    overlay[mask > 0] = color
                    # Blend with original frame
                    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # Draw bounding boxes
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Box color matches mask color
                    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
                    color = colors[i % len(colors)]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with score
                    if i < len(scores):
                        label = f"{self.text_prompt}: {scores[i]:.2f}"
                        
                        # Add background for text
                        text_size = cv2.getTextSize(
                            label, self._font, self._font_scale, self._font_thickness
                        )[0]
                        cv2.rectangle(
                            frame,
                            (x1, y1 - text_size[1] - 10),
                            (x1 + text_size[0], y1),
                            color,
                            -1
                        )
                        
                        cv2.putText(
                            frame, label, (x1, y1 - 5),
                            self._font, self._font_scale, (0, 0, 0), self._font_thickness
                        )

        except Exception as e:
            logger.warning(f"⚠️ Annotation failed: {e}")

        return frame
